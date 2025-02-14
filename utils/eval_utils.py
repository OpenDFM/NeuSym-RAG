#coding=utf8
import ast, json, math, os, re, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from fuzzywuzzy import fuzz, process
from tabulate import tabulate
from contextlib import nullcontext
from evaluation.evaluator import evaluate_airqa
from agents.models import get_llm_single_instance
from utils.functions.common_functions import is_valid_uuid


def evaluate_dataset(dataset: str, pred_ans: str, gold_data: Dict[str, Any], **kwargs) -> float:
    """ Given the dataset name and question type, evaluate whether the predicted answer is consistent with the gold data (only comparing one data point).
    @args:
        dataset: str, dataset name
        pred_ans: str, predicted answer
        gold_data: Dict[str, Any], gold data
    @return:
        score: float, evaluation score
    """

    if dataset == 'pdfvqa':
        score = evaluate_pdfvqa(pred_ans, gold_data, **kwargs)
    elif dataset == 'tatdqa':
        score = evaluate_tatdqa(pred_ans, gold_data, **kwargs)
    elif dataset in ['airqa', 'm3sciqa', 'scidqa', 'spiqa']:
        score = evaluate_airqa(pred_ans, gold_data)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")
    return score


def llm_semantic_equivalent(question: str, pred_ans: str, gold_ans: str, model: str = 'gpt-4o', temperature: float = 0.7, top_p: float = 0.95) -> float:
    prompt = """You are given the following question and answer pair, please determine whether the predicted answer is semantically-equivalent to the gold answer. Your response should be a single integer number 0 or 1, with 1 for equivalent and 0 for not equivalent (no punctuation and formatting).
Question: {question}
Predicted Answer: {pred_ans}
Gold Answer: {gold_ans}
Response: 
""".format(question=question, pred_ans=pred_ans, gold_ans=gold_ans)
    messages = [{'role': 'user', 'content': prompt}]
    model_client = get_llm_single_instance(model)
    response = model_client.get_response(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p
    )
    try:
        score = float(response)
    except Exception as e:
        score = 0.0
    return score


def fuzzy_match_strs(question: str, pred_ans: str, gold_ans: str, threshold: float = 0.95) -> float:
    normalize = lambda x: re.sub(r'\s+', ' ', str(x).strip())
    q, pred, gold = normalize(question), normalize(pred_ans), normalize(gold_ans)
    # allow pred to fully contain the gold answer (treated as true)
    compare_function = fuzz.partial_ratio if len(pred) > len(gold) else fuzz.ratio
    return 1.0 if compare_function(pred, gold) / 100.0 >= threshold else llm_semantic_equivalent(q, pred, gold)


def fuzzy_match_lists(question: str, pred_ans: List[str], gold_ans: List[str], threshold: float = 0.95, require_order: int = 0) -> float:
    if require_order:
        if len(pred_ans) != len(gold_ans):
            return 0.0
        for i in range(len(pred_ans)):
            if fuzzy_match_strs(question, pred_ans[i], gold_ans[i], threshold=threshold) < threshold:
                return 0.0
        return 1.0
    else:
        gold_ans_copy = gold_ans[:]
        for ans in pred_ans:
            if not gold_ans_copy:
                return 0.0
            best_match = process.extractOne(
                ans,
                gold_ans_copy,
                scorer = lambda x, y: fuzzy_match_strs(question, x, y, threshold=threshold) * 100.0
            )
            if best_match and best_match[1] / 100.0 >= threshold:
                gold_ans_copy.remove(best_match[0])
            else:
                return 0.0
        return 1.0


def extract_list_from_str(s: str):
    match = re.search(r'(\[.*\])', s)
    if match:
        list_str = match.group(0)
        try:
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            return s
    return s


def evaluate_pdfvqa(pred_ans: Union[List[str], str], gold_data: Dict[str, Any], **kwargs) -> float:
    """ Evaluate the predicted answer for the PDFVQA dataset.
    @args:
        pred_ans: str, predicted answer
        gold_data: Dict[str, Any], gold data
    @return:
        score: float, evaluation score
    """
    question, question_type = gold_data['question'], gold_data['question_type']
    assert question_type in ['existence', 'counting', 'object_recognition', 'structural_understanding', 'parent_relationship_understanding', 'child_relationship_understanding']

    def exact_match(pred_ans: str, gold_ans: str) -> float:
        return float(pred_ans == gold_ans)

    if question_type in ['existence', 'counting']:
        answer = str(gold_data['answer']).strip().lower()
        pred_ans = str(pred_ans).strip().lower()
        return exact_match(pred_ans, answer)
    elif question_type in ['object_recognition', 'structural_understanding']: # mostly verbose section titles or recognized sentence, or special answer "No specific Section."
        threshold = kwargs.pop('threshold', 0.95)
        answer = str(gold_data['answer']).strip().lower()
        pred_ans = str(pred_ans).strip().lower()
        return fuzzy_match_strs(question, pred_ans, answer, threshold=threshold)
    elif question_type in ['parent_relationship_understanding', 'child_relationship_understanding']:
        threshold = kwargs.pop('threshold', 0.95)
        pred_ans = extract_list_from_str(str(pred_ans))
        if not isinstance(pred_ans, list):
            pred_ans = [pred_ans]
        return fuzzy_match_lists(question, pred_ans, gold_data['answer'], threshold=threshold)
    else:
        raise NotImplementedError(f"Question type {question_type} not supported.")


def evaluate_tatdqa(pred_ans: Any, gold_data: Dict[str, Any], **kwargs) -> float:
    """ Evaluate the predicted answer for the TATDQA dataset.
    @args:
        pred_ans: LLM predicted str, predicted answer
        gold_data: Dict[str, Any], gold data containing 'uuid', 'question', 'question_type', 'answer', etc.
    """

    question, question_type, gold_scale, gold_answer = gold_data['question'], gold_data['question_type'], gold_data['answer'][1], gold_data['answer'][0]
    pred_ans = extract_list_from_str(str(pred_ans))
    if gold_scale == '' and question_type != 'arithmetic' or not isinstance(pred_ans, list):
        pred_ans = [pred_ans, '']
    if len(pred_ans) != 2:
        return 0.0
    pred_answer, pred_scale = pred_ans[0], pred_ans[1]

    if pred_scale not in ['percent', 'thousand', 'million', '']:
        return 0.0
    assert question_type in ['span', 'multi-span', 'arithmetic', 'count']

    def to_float(gold_ans: Any) -> float:
        allowed_characters = "0123456789-."
        gold_ans = str(gold_ans).strip().lower()
        gold_ans = ''.join([ch for ch in gold_ans if ch in allowed_characters])
        try:
            return float(gold_ans)
        except ValueError:
            return math.nan

    def exact_match(pred_ans: Any, gold_ans: Any) -> float:
        pred_ans = to_float(pred_ans)
        gold_ans = to_float(gold_ans)
        if math.isnan(pred_ans) or math.isnan(gold_ans):
            return 0.0
        return float(round(pred_ans) == round(gold_ans))

    if question_type in ['arithmetic', 'count']:
        return exact_match(pred_answer, gold_answer) * float(gold_scale == str(pred_scale))
    elif question_type in ['span', 'multi-span']:
        if gold_scale == '':
            threshold = kwargs.pop('threshold', 0.95)
            if question_type == 'span':
                pred_answer = [str(pred_answer)]
            return fuzzy_match_lists(question, pred_answer, gold_answer, threshold=threshold, require_order=1)
        elif gold_scale in ['percent', 'thousand', 'million']:
            threshold = kwargs.pop('threshold', 0.95)
            if question_type == 'multi-span':
                if len(pred_answer) != len(gold_answer):
                    return 0.0
                for i in range(len(pred_answer)):
                    if exact_match(pred_answer[i], gold_answer[i]) < threshold:
                        return 0.0
                return float(gold_scale == str(pred_scale))
            else:
                return exact_match(str(pred_answer), gold_answer) * float(gold_scale == str(pred_scale))
        else:
            raise NotImplementedError(f"Gold scale {gold_scale} not supported.")
    else:
        raise NotImplementedError(f"Question type {question_type} not supported.")


def resolve_gold_answer(gold: Dict[str, Any]) -> str:
    if 'answer' in gold:
        return gold['answer']
    assert 'evaluator' in gold and 'eval_kwargs' in gold['evaluator'], f"Please check the data format for {gold['uuid']}."
    if 'answer' in gold['evaluator']['eval_kwargs']:
        return gold['evaluator']['eval_kwargs']['answer']
    # raise ValueError(f"Gold answer not found in evaluator: {str(gold['evaluator'])}.")
    return str(json.dumps(gold['evaluator'], ensure_ascii=False))


def evaluate(pred: Union[List[dict], str], gold: Union[List[dict], str], dataset: str, **kwargs) -> dict:
    """ Evaluate the predicted answers on the entire dataset. Note that,
    both the gold and predict answers are stored in jsonl format.
    @args:
        pred: Union[List[dict], str], JSONL path to predicted answer or JSON list
        gold: Union[List[dict], str], JSONL path to gold answer or JSON list
        dataset: str, dataset name
        kwargs: dict, additional arguments, e.g.,
            output_path: str, path to save the evaluation result
            threshold: float, threshold for fuzzy matching for pdfvqa and tatdqa, default 0.95
    @return:
        result: dict, each key contains the count, correct count, and score. The special key 'all' contains the overall evaluation score, e.g.,
            {
                "all": {"score": 0.8, "count": 100, "correct": 80},
                "existence": {"score": 0.9, "count": 20, "correct": 18},
                "counting": {"score": 0.7, "count": 20, "correct": 14},
                ...
            }
    """

    pred_data, gold_data = [], []
    if isinstance(pred, str):
        with open(pred, 'r', encoding='utf8') as f:
            for line in f:
                pred = json.loads(line)
                pred_data.append(pred)
    else: pred_data = pred
    if isinstance(gold, str):
        with open(gold, 'r', encoding='utf8') as f:
            for line in f:
                gold = json.loads(line)
                gold_data.append(gold)
    else: gold_data = gold

    assert len(pred_data) == len(gold_data)
    result = defaultdict(lambda: {'score': 0.0, 'count': 0, 'correct': 0}) # (score, count)
    output_path = kwargs.get('output_path', None)
    with open(output_path, 'w', encoding='utf-8') if output_path else nullcontext() as outfile:
        for pred, gold in zip(pred_data, gold_data):
            score = evaluate_dataset(dataset, pred['answer'], gold, **kwargs)
            if score < 0.5 and output_path is not None:
                outfile.write(f'\n[ERROR]: data (type={gold["question_type"] if "question_type" in gold else gold["tags"]}) with id {gold["uuid"]}\n')
                outfile.write(f'Gold Answer: {resolve_gold_answer(gold)}\n')
                outfile.write(f'Predicted Answer: {pred["answer"]}\n')
            result['all']['count'] += 1
            result['all']['correct'] += score
            if 'question_type' in gold:
                result[gold['question_type']]['count'] += 1
                result[gold['question_type']]['correct'] += score
            else:
                for tag in gold['tags']:
                    result[tag]['count'] += 1
                    result[tag]['correct'] += score

        for key in result.keys():
            score, count = result[key]['correct'], result[key]['count']
            result[key]['score'] = score / count if count > 0 else 0.0

        if output_path is not None:
            outfile.write('\n' + print_result(result))

    return result


def re_evaluate(pred: Union[List[dict], str], gold: Union[List[dict], str], eval_path: str, dataset: str, **kwargs) -> dict:
    """ Re-evaluate AirQA dataset
    """

    # if dataset != 'airqa':
    #     raise NotImplementedError(f"Dataset {dataset} not supported.")
    
    pred_data, gold_data = [], []
    if isinstance(pred, str):
        with open(pred, 'r', encoding='utf8') as f:
            for line in f:
                pred = json.loads(line)
                pred_data.append(pred)
    else: pred_data = pred
    if isinstance(gold, str):
        with open(gold, 'r', encoding='utf8') as f:
            for line in f:
                gold = json.loads(line)
                gold_data.append(gold)
    else: gold_data = gold
    
    error_uuids = []
    block_lines = ""
    result_lines = ""
    with open(eval_path, 'r', encoding='utf8') as f:
        text = ""
        is_block = True
        for line in f:
            if line.startswith("+") and line.strip().endswith("+"):
                is_block = False
            if is_block: 
                block_lines += line
            else:
                result_lines += line

    block_lines += "\n[ERROR]\n"
    
    for block in block_lines.split('[ERROR]: data'):
        block = block.strip()
        if not block: continue
        first_line = block.split('\n')[0]
        uuid = first_line.split('with id ')[-1]
        if is_valid_uuid(uuid):
            error_uuids.append(uuid)
        else:
            print(f"Invalid block: {block}")
    
    print(len(error_uuids))
    
    result_lines = list(result_lines.strip().split("\n"))
    result_dict = {}
    for line in result_lines[3:-1]:
        values = [value.strip() for value in line[1:-1].split("|")]
        result_dict[values[0]] = [float(values[1]), int(values[2]), float(values[3])]
        assert abs(float(values[1]) / int(values[2]) - float(values[3])) < 1e-6, f"Error in Original Result data:\n{line}"
    
    print(result_dict)

    assert len(pred_data) == len(gold_data)
    result = defaultdict(lambda: {'score': 0.0, 'count': 0, 'correct': 0}) # (score, count)
    output_path = kwargs.get('output_path', None)
    with open(output_path, 'w', encoding='utf-8') if output_path else nullcontext() as outfile:
        for pred, gold in zip(pred_data, gold_data):
            score = evaluate_dataset(dataset, pred['answer'], gold, **kwargs)
            if score < 0.5 and output_path is not None:
                outfile.write(f'\n[ERROR]: data (type={gold["question_type"] if "question_type" in gold else gold["tags"]}) with id {gold["uuid"]}\n')
                outfile.write(f'Gold Answer: {resolve_gold_answer(gold)}\n')
                outfile.write(f'Predicted Answer: {pred["answer"]}\n')
            result['all']['count'] += 1
            result['all']['correct'] += score
            if 'question_type' in gold:
                result[gold['question_type']]['count'] += 1
                result[gold['question_type']]['correct'] += score
            else:
                for tag in gold['tags']:
                    result[tag]['count'] += 1
                    result[tag]['correct'] += score

        for key in result.keys():
            score, count = result[key]['correct'], result[key]['count']
            result[key]['score'] = score / count if count > 0 else 0.0

        if output_path is not None:
            outfile.write('\n' + print_result(result))

    return result


def print_result(result: dict) -> str:
    """ Print the evaluation result.
    @args:
        result: dict, evaluation result
    """
    table_data = [[key, values['correct'], values['count'], values['score']] for key, values in result.items() if key != 'all']
    table_data += [['total', result['all']['correct'], result['all']['count'], result['all']['score']]]
    headers = ["Question Type", "Correct", "Total", "Score"]
    return tabulate(table_data, headers=headers, tablefmt='pretty')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='airqa', choices=['airqa', 'pdfvqa', 'tatdqa', 'm3sciqa', 'scidqa', 'spiqa'], help='Dataset name')
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted answer, .jsonl file')
    parser.add_argument('--gold', type=str, required=True, help='Path to gold answer, .jsonl file')
    parser.add_argument('--output', type=str, required=False, default=None, help='Path to save the evaluation result, .txt file')
    parser.add_argument('--eval', type=str, required=False, default=None, help='Path to previous evaluation results, .txt file')
    args = parser.parse_args()
    
    if args.eval is not None:
        result = re_evaluate(args.pred, args.gold, args.eval, args.dataset, output_path=args.output)
    else:
        result = evaluate(args.pred, args.gold, args.dataset, output_path=args.output)
    if result:
        result_table = print_result(result)
        print(f"Final evaluation result on {args.dataset}:\n{result_table}")
#coding=utf8
import json, math, os, re, sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from fuzzywuzzy import fuzz
from tabulate import tabulate
from contextlib import nullcontext
from evaluation.evaluator import evaluate_airqa as evaluate_dataset
from utils.functions.common_functions import is_valid_uuid


def load_jsonl(fp: str) -> List[Dict[str, Any]]:
    with open(fp, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    return


def load_test_data(test_data: str, dataset: str = 'ariqa') -> List[Dict[str, Any]]:
    examples = []
    if os.path.exists(test_data) and os.path.isfile(test_data):
        test_data_path = test_data
    else:
        test_data_path = os.path.join('data', 'dataset', dataset, test_data)
        if not os.path.exists(test_data_path):
            raise ValueError('[ERROR]: Filepath for test data {} not found.'.format(test_data_path))
    examples = load_jsonl(test_data_path)
    return examples


def resolve_gold_answer(gold: Dict[str, Any]) -> str:
    assert 'evaluator' in gold and 'eval_kwargs' in gold['evaluator'], f"Please check the data format for {gold['uuid']}."
    for k in ['answer', 'reference', 'reference_answer', 'gold']:
        if k in gold['evaluator']['eval_kwargs']:
            return gold['evaluator']['eval_kwargs'][k]
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
    @return:
        result: dict, each key contains the count, correct count, and score. The special key 'all' contains the overall evaluation score, e.g.,
            {
                "all": {"score": 0.8, "count": 100, "correct": 80},
                "text": {"score": 0.9, "count": 20, "correct": 18},
                "image": {"score": 0.7, "count": 20, "correct": 14},
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
        cnt, tot = 0, len(pred_data)
        for pred, gold in zip(pred_data, gold_data):
            cnt += 1
            print(f"Evaluating {cnt}/{tot}...", end='\r')
            score = evaluate_dataset(pred['answer'], gold)
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

    if dataset != 'airqa':
        raise NotImplementedError(f"Dataset {dataset} not supported.")
    
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
    
    result_lines = list(result_lines.strip().split("\n"))
    result = {}
    for line in result_lines[3:-1]:
        values = [value.strip() for value in line[1:-1].split("|")]
        result[values[0]] = {
            "correct": float(values[1]), 
            "count": int(values[2]), 
            "score": float(values[3])
        }
        assert abs(float(values[1]) / int(values[2]) - float(values[3])) < 1e-6, f"Error in Original Result data:\n{line}"
    
    if result.get('figure'):
        result['image']['correct'] += result['figure']['correct']
        result['image']['count'] += result['figure']['count']
        del result['figure']
    result['all'] = result['total']
    del result['total']

    tot = 0
    assert len(pred_data) == len(gold_data)
    output_path = kwargs.get('output_path', None)
    with open(output_path, 'w', encoding='utf-8') if output_path else nullcontext() as outfile:
        for pred, gold in tqdm(zip(pred_data, gold_data)):
            # Only Re-Evaluate LitSearch
            if (gold["annotator"] == 'human') or ("retrieval" not in gold["tags"]):
                continue
            
            tot += 1
            assert gold["evaluator"]["eval_func"] == "eval_paper_relevance_with_llm_and_reference_answer", f"Eval Func `eval_paper_relevance_with_llm_and_reference_answer` expected, but {gold['evaluator']['eval_func']} found."
            
            pre_score = 0.0 if gold["uuid"] in error_uuids else 1.0
            result['all']['count'] -= 1
            result['all']['correct'] -= pre_score
            if 'question_type' in gold:
                result[gold['question_type']]['count'] -= 1
                result[gold['question_type']]['correct'] -= pre_score
            else:
                for tag in gold['tags']:
                    result[tag]['count'] -= 1
                    result[tag]['correct'] -= pre_score
            
            assert ('subjective' in gold['tags']) and ('objective' not in gold['tags'])
            gold['tags'].remove('subjective')
            gold['tags'].append('objective')
            
            reference_answer = str(gold["evaluator"]["eval_kwargs"]["reference_answer"]).lower()
            pred_answer = str(pred['answer']).lower()
            score = 0.0
            if fuzz.ratio(pred_answer, reference_answer) >= 90:
                score = 1.0
            
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

    print(f"Re-Evaluate {tot} examples.")
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
    parser.add_argument('--dataset', type=str, default='airqa', choices=['airqa', 'm3sciqa', 'scidqa', 'spiqa'], help='Dataset name')
    parser.add_argument('--pred', type=str, help='Path to predicted answer, .jsonl file')
    parser.add_argument('--gold', type=str, help='Path to gold answer, .jsonl file')
    # parser.add_argument('--folder', type=str, required=True, help='Folder to results & evaluations.')
    # parser.add_argument('--re', action='store_true', help='Whether to re-evaluate the results')
    parser.add_argument('--output', type=str, default=None, help='Path to save the evaluation result, .txt file')
    args = parser.parse_args()
    
    # folder = args.folder
    # assert os.path.exists(folder), "[Error]: Folder not found."
    # result_path = os.path.join(folder, 'result.jsonl')
    # assert os.path.exists(result_path), "[Error]: Result file not found."
    # if args.re:
    #     eval_path = os.path.join(folder, 'evaluation.txt')
    #     assert os.path.exists(eval_path), "[Error]: Eval file not found."
    #     output_path = os.path.join(folder, 're_evaluation.txt')
    #     result = re_evaluate(result_path, args.gold, eval_path, args.dataset, output_path=output_path)
    #     exit(0)

    assert args.pred, "[Error]: Path to predicted answer .jsonl is required."
    assert args.gold, "[Error]: Path to gold answer .jsonl is required."
    result = evaluate(args.pred, args.gold, args.dataset, output_path=args.output_path)
    result_table = print_result(result)
    print(f"Final evaluation result on {args.dataset}:\n{result_table}")
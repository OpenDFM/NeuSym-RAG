#coding=utf8
import os, openai, re
from typing import Dict, Any, Optional, List, Union
from fuzzywuzzy import fuzz


def evaluation(dataset: str, pred_ans: str, gold_data: Dict[str, Any], **kwargs) -> float:
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
        pass
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")
    return score


def llm_semantic_equivalent(question: str, pred_ans: str, gold_ans: str, model: str = 'gpt-4o', temperature: float = 0.7, top_p: float = 0.9) -> float:
    prompt = """You are given the following question and answer pair, please determine whether the predicted answer is semantically-equivalent to the gold answer. Your response should be a single integer number 0 or 1, with 1 for equivalent and 0 for not equivalent (no punctuation and formatting).
Question: {question}
Predicted Answer: {pred_ans}
Gold Answer: {gold_ans}
Response: 
""".format(question=question, pred_ans=pred_ans, gold_ans=gold_ans)
    messages = [{'role': 'user', 'content': prompt}]
    if os.environ.get('OPENAI_BASE_URL', None) is not None:
        openai.base_url = os.environ['OPENAI_BASE_URL']
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
        temperature=temperature
    )
    result = response.choices[0].message.content.strip()
    try:
        score = float(result)
    except Exception as e:
        score = 0.0
    return score


def evaluate_pdfvqa(pred_ans: Union[List[str], str], gold_data: Dict[str, Any], question_type: Optional[str] = None, **kwargs) -> float:
    """ Evaluate the predicted answer for the PDFVQA dataset.
    @args:
        pred_ans: str, predicted answer
        gold_data: Dict[str, Any], gold data
        question_type: str, question type, optional
    @return:
        score: float, evaluation score
    """
    question_type = gold_data['question_type']
    assert question_type in ['existence', 'counting', 'object_recognition', 'structural_understanding', 'parent_relationship_understanding', 'child_relationship_understanding']

    def exact_match(pred_ans: str, gold_ans: str) -> float:
        return float(pred_ans == gold_ans)
    
    def fuzzy_match(pred_ans: str, gold_ans: str, threshold: float = 0.95) -> float:
        return float(fuzz.ratio(re.sub(r'\s+', ' ', pred_ans), re.sub(r'\s+', ' ', gold_ans)) / 100.0 >= threshold)

    if question_type in ['existence', 'counting']:
        answer = gold_data['answer'].strip().lower()
        pred_ans = pred_ans.strip().lower()
        return exact_match(pred_ans, answer)
    elif question_type in ['object_recognition', 'structural_understanding']: # mostly verbose section titles or recognized sentence, or special answer "No specific Section."
        threshold = kwargs.pop('threshold', 0.95)
        answer = gold_data['answer'].strip().lower()
        pred_ans = pred_ans.strip().lower()
        return fuzzy_match(pred_ans, answer, threshold=threshold)
    elif question_type in ['parent_relationship_understanding', 'child_relationship_understanding']:
        model, temperature, top_p = kwargs.get('model', 'gpt-4o'), kwargs.get('temperature', 0.7), kwargs.get('top_p', 0.95)
        return llm_semantic_equivalent(gold_data['question'], pred_ans, gold_data['answer'], model=model, temperature=temperature, top_p=top_p)
    else:
        raise NotImplementedError(f"Question type {question_type} not supported.")


def evaluate(pred: Union[List[dict], str], gold: Union[List[dict], str], dataset: str, **kwargs) -> float:
    """ Evaluate the predicted answers on the entire dataset. Note that,
    both the gold and predict answers are stored in jsonl format.
    @args:
        pred: Union[List[dict], str], JSONL path to predicted answer or JSON list
        gold: Union[List[dict], str], JSONL path to gold answer or JSON list
        dataset: str, dataset name
    @return:
        score: float, evaluation score
    """

    pred_data, gold_data = [], []
    if isinstance(pred, str):
        with open(pred, 'r') as f:
            for line in f:
                pred = json.loads(line)
                pred_data.append(pred)
    else: pred_data = pred
    if isinstance(gold, str):
        with open(gold, 'r') as f:
            for line in f:
                gold = json.loads(line)
                gold_data.append(gold)
    else: gold_data = gold

    assert len(pred_data) == len(gold_data)
    total_score, count = 0.0, 0
    verbose = kwargs.pop('verbose', True)
    for pred, gold in zip(pred_data, gold_data):
        score = evaluation(dataset, pred['answer'], gold, **kwargs)
        if score < 0.5 and verbose:
            print(f'[ERROR]: data with id {gold["uuid"]}')
        total_score += score
        count += 1
    if verbose:
        print(f"Total evaluation score on {dataset}: {total_score / count:.4f}")
    
    return total_score / count


if __name__ == '__main__':

    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pdfvqa', choices=['pdfvqa', 'tatdqa'], help='Dataset name')
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted answer')
    parser.add_argument('--gold', type=str, required=True, help='Path to gold answer')
    args = parser.parse_args()

    evaluate(args.pred, args.gold, args.dataset)
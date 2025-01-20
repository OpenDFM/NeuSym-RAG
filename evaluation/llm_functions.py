#coding=utf8
import re, json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Any, Dict, List, Tuple, Optional
from utils.functions.common_functions import call_llm, call_llm_with_message
from utils.airqa_utils import get_relevent_papers_by_title


DEFAULT_LLM_MODEL = 'gpt-4o' # may be changed to other open-source models
DEFAULT_TEMPERATURE = 0.0


def _eval_with_llm(template, llm_model, temperature) -> float:
    # Call the LLM model
    llm_output = call_llm(template, llm_model, temperature=temperature)
    # Extract the final judgement
    matched = re.findall(r'```(txt)?\s(.*?)\s```', llm_output, re.DOTALL)
    if len(matched) == 0:
        return 0.0
    final_judgement = matched[-1][1].strip().lower()
    # Return the final judgement
    return 1.0 if final_judgement == 'true' else 0.0


def eval_reference_answer_with_llm(pred: Any, reference_answer: str, question: str, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> float:
    """ Evaluate the reference answer with LLM.
    @param:
        pred: The predicted answer.
        reference_answer: The reference answer.
        question: The input question.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    # Prepare the input
    template = f"""You are an intelligent judgement system who is expert in determining whether a predicted answer matches the reference answer in terms of semantic meaning and intent, based on the input question. You will be given the raw question, the reference answer, and the predicted answer. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The final decision string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The structure and format of the predicted answer do not matter. We only care about the semantic content, compared to the reference answer. Minor differences in grammar, structure, or formatting should be ignored if the core meaning is preserved.
Now, let's start!

[Question]: {question}
[Reference Answer]: {reference_answer}
[Predicted Answer]: {str(pred)}

Let's think step-by-step, and then provide the final judgement.
"""
    return _eval_with_llm(template, llm_model, temperature)


def eval_candidate_reference_answer_with_llm(pred: Any, candidate_reference_answers: List[str], question: str, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> float:
    """ Evaluate the candidate reference answers with LLM. The predicted answer should be considered correct if any of the candidate reference answers is matched semantically.
    @param:
        pred: The predicted answer.
        candidate_reference_answers: The list of candidate reference answers.
        question: The input question.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    # Prepare the input
    num_candidate_reference_answers = len(candidate_reference_answers)
    candidate_reference_answers_str = '\n\n'.join([f'## Candidate Reference Answer {idx + 1}\n' + cra.strip() for idx, cra in enumerate(candidate_reference_answers)])
    template = f"""You are an intelligent judgement system who is expert in determining whether a predicted answer matches anyone of the candidate reference answers in terms of semantic meaning and intent, based on the input question. You will be given the raw question, all candidate reference answers, and the predicted answer. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The final decision string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The structure and format of the predicted answer do not matter. We only care about the semantic content, compared to the candidate reference answers. Minor differences in grammar, structure, or formatting should be ignored if the core meaning is preserved.
4. The predicted answer is considered correct IF AND ONLY IF it matches anyone of the candidate reference answers.
Now, let's start!

[Question]: {question}
[Candidate Reference Answers]: In total, there are {num_candidate_reference_answers} candidate reference answers:
{candidate_reference_answers_str}

[Predicted Answer]: {str(pred)}

Let's think step-by-step, and then provide the final judgement.
"""
    return _eval_with_llm(template, llm_model, temperature)


def eval_partial_scoring_points_with_llm(pred: Any, scoring_points: List[str], question: str, count: int = 1, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> float:
    """ Evaluate whether the scoring points are partially mentioned in the pred answer with LLM (at least `count`).
    @param:
        pred: The predicted answer.
        scoring_points: The list of scoring points.
        question: The input question.
        count: int, the minimum number of scoring points that must be mentioned in the predicted answer.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    assert count > 0 and count < len(scoring_points), f"The count must be greater than 0 and less than the total number {len(scoring_points)} of given scoring points."
    # Prepare the input
    num_scoring_points = len(scoring_points)
    scoring_points_str = '\n'.join(['- ' + sp.strip() for sp in scoring_points])
    template = f"""You are an intelligent judgement system who is expert in determining whether a predicted answer mentions each of the scoring points for the input question. You will be given the raw question, all scoring points, and the predicted answer. The predicted answer is considered correct IF AND ONLY IF at least {count} scoring point{' is' if count == 1 else 's are'} mentioned. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The final decision string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The predicted answer is only considered as correct IF AND ONLY IF the predicted answer correctly mentions at least {count} scoring point{'' if count == 1 else 's'}.
4. The structure, format and order of the scoring points do not matter. We only care about the semantics and content.
Now, let's start!

[Question]: {question}
[Scoring points]: In total, there are {num_scoring_points} scoring points and the predicted answer needs to mention at least {count} of them:
{scoring_points_str}
[Predicted Answer]: {str(pred)}

Let's think step-by-step, and then provide the final judgement.
"""
    return _eval_with_llm(template, llm_model, temperature)


def eval_scoring_points_with_llm(pred: Any, scoring_points: List[str], question: str, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, ignore_order: bool = True, **kwargs) -> float:
    """ Evaluate whether the scoring points are ALL mentioned in the pred answer with LLM.
    @param:
        pred: The predicted answer.
        scoring_points: The list of scoring points.
        question: The input question.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
        ignore_order: Whether to ignore the order of scoring points.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    # Prepare the input
    num_scoring_points = len(scoring_points)
    scoring_points_str = '\n'.join(['- ' + sp.strip() for sp in scoring_points])
    template = f"""You are an intelligent judgement system who is expert in determining whether a predicted answer exactly mentions all required scoring points for the input question. You will be given the raw question, all required scoring points, and the predicted answer. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The final decision string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The predicted answer is only considered as correct IF AND ONLY IF the required scoring points are ALL mentioned in the predicted answer.
4. The {'structure, format and order' if ignore_order else 'structure and format'} of the scoring points do not matter. We only care about the semantics and content.
Now, let's start!

[Question]: {question}
[Scoring points]: In total, there are {num_scoring_points} scoring points:
{scoring_points_str}
[Predicted Answer]: {str(pred)}

Let's think step-by-step, and then provide the final judgement.
"""
    return _eval_with_llm(template, llm_model, temperature)


def eval_reference_answer_and_scoring_points_with_llm(pred: Any, reference_answer: str, scoring_points: List[str], question: str, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, ignore_order: bool = True, **kwargs) -> float:
    """ Evaluate the reference answer and scoring points with LLM.
    @param:
        pred: The predicted answer.
        reference_answer: The reference answer.
        scoring_points: The list of scoring points.
        question: The input question.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
        ignore_order: Whether to ignore the order of scoring points.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    # Prepare the input
    num_scoring_points = len(scoring_points)
    scoring_points_str = '\n'.join(['- ' + sp.strip() for sp in scoring_points])
    template = f"""You are an intelligent judgement system who is expert in determining whether a predicted answer matches the reference answer and mentions all required scoring points in terms of semantic meaning and intent, based on the input question. You will be given the raw question, the reference answer, all required scoring points, and the predicted answer. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The final decision string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The predicted answer is only considered as correct ONLY IF the required scoring points are ALL mentioned in the predicted answer.
4. The {'structure, format and order' if ignore_order else 'structure and format'} of the scoring points do not matter. We only care about the semantics and content.
5. For comparison with the reference answer, minor differences in grammar, structure, or formatting should be ignored if the core meaning is preserved.
Now, let's start!

[Question]: {question}
[Reference Answer]: {reference_answer}
[Scoring Points]: In total, there are {num_scoring_points} scoring points:
{scoring_points_str}
[Predicted Answer]: {str(pred)}
    """
    return _eval_with_llm(template, llm_model, temperature)


def eval_paper_relevance_with_llm(pred: Any, question: str, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """Evaluate the relevance of the predicted paper with the question.
    @param:
        pred: The predicted paper title.
        question: The input question.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    results = get_relevent_papers_by_title(pred)
    if len(results) == 0:
        return 0.0
    metadata = results[0]
    title, abstract, authors, conference, year, volume = metadata["title"], metadata["abstract"], metadata["authors"], metadata["conference"], metadata["year"], metadata["volume"]
    template = f"""You are an intelligent judgement system who is expert in determining whether the predicted paper matches the given question. You will be given the metadata of the paper, and the original question. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The final decision string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The predicted paper is considered as correct IF AND ONLY IF the title and abstract of it matches the description of the question.
Now, let's start!

[Question]: {question}
[Title]: {title}
[Conference]: {conference}
[Year]: {year}
[Volume]: {volume}
[Authors]: {", ".join(authors)}
[Abstract]: {abstract}
"""
    return _eval_with_llm(template, llm_model, temperature)
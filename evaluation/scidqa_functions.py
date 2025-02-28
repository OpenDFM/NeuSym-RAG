#coding=utf8
from typing import Any
try:
    from utils.functions.common_functions import call_llm_with_message
except:
    import openai, os
    from openai.types.chat.chat_completion import ChatCompletion

    def call_llm_with_message(
            messages: Any, 
            model: str = 'gpt-4o-mini', 
            top_p: float = 0.95, 
            temperature: float = 0.7
        ) -> str:
        """ Call LLM to generate the response directly using the message list.
        """
        api_key = os.getenv('OPENAI_API_KEY', None)
        base_url = os.getenv('OPENAI_BASE_URL', None)
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        completion: ChatCompletion = client.chat.completions.create(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p
        )
        return completion.choices[0].message.content.strip()


DEFAULT_SCIDQA_LLM_MODEL = 'gpt-4o-mini' # too many examples, thus cheaper models
DEFAULT_SCIDQA_TEMPERATURE = 0.1


SCIDQA_INSTRUCTION = """You are an expert evaluator tasked with assessing the quality of a model-generated answer compared to a gold standard correct answer in a long-form question-answering context. Your goal is to provide a quantified evaluation across multiple dimensions. Please follow these steps:

Carefully read the original question, the model-generated answer, and the gold correct answer. Evaluate the model-generated answer on the following dimensions, providing a score from 1-10 for each (where 1 is poor and 10 is excellent): a) Relevance (1-10): How well does the answer address the specific question asked? b) Accuracy (1-10): To what extent is the information provided correct and aligned with the gold answer? c) Completeness (1-10): How thoroughly does the answer cover all aspects of the question compared to the gold answer? d) Conciseness (1-10): Does the answer provide information efficiently without unnecessary details?

Calculate an overall quality score by taking the average of the five dimension scores. In your answer for each dimension, provide a justification why not a higher score and why not a lower score.

Question: {}

Model-generated Answer: {}

Gold Correct Answer: {}

Structure your response as follows:

Evaluation:
1. Relevance: [Score] - [Explanation]
2. Accuracy: [Score] - [Explanation]
3. Completeness: [Score] - [Explanation]
4. Conciseness: [Score] - [Explanation]

Overall Quality Score: [Average of the four above scores]"""


SCIDQA_EXTRACT_RESULT_INSTRUCTION = """You are provided with an evaluation of an answer in the following format:
    
Evaluation:
1. Relevance: [Score] - [Explanation]
2. Accuracy: [Score] - [Explanation]
3. Completeness: [Score] - [Explanation]
4. Conciseness: [Score] - [Explanation]
Overall Quality Score: [Average of the four above scores].

Carefully read the evaluation provided next, and extract the final overall quality score from the discussion. Do not include any explanation, you should only provide the final numeric score for overall quality from the evaluation statement.
"""


def eval_scidqa(pred: Any, reference_answer: str = '', question: str = '', model: str = DEFAULT_SCIDQA_LLM_MODEL, temperature: float = DEFAULT_SCIDQA_TEMPERATURE) -> float:
    try:
        llm_pred = str(pred)
        gt = str(reference_answer).strip()
        if gt.startswith("A: "):
            gt = gt[3:]
            gt = gt.strip()
        local_instr = SCIDQA_INSTRUCTION.format(question, llm_pred, gt)
        response = call_llm_with_message([{'role': 'user', 'content': local_instr}], model=model, temperature=temperature, top_p=0.9)

        # Extract the evaluation score
        final_instr = SCIDQA_EXTRACT_RESULT_INSTRUCTION + '\n\n' + response
        score = call_llm_with_message([{'role': 'user', 'content': final_instr}], model=model, temperature=temperature, top_p=0.9)
        score = round(float(score), 2)
    except Exception as e:
        # print(e)
        score = 0.
        print('[ERROR]: Unexpected error occurred during the evaluation of SCIDQA dataset.')
    return score
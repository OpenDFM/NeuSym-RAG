#coding=utf8
"""LLM Log-Likelihood Scoring for OpenAI GPT models.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any
from .eval_spiqa.llmlogscore import OpenAIClient as SpiQAEvalClient
from .match_functions import eval_bool_exact_match


_SUFFIXES_TO_SCORE = [' yes', ' yeah']
_COMPLEMENT_SUFFIXES = [' no']
_SPIQA_EVAL_CLIENT = {}

DEFAULT_SPIQA_LLM_MODEL = 'gpt-4o-mini' # too many examples, thus cheaper models
DEFAULT_SPIQA_TEMPERATURE = 0.0


_PROMPT = 'You are given a question, ground-truth answer, and a candidate answer. Question: <question> \nGround-truth answer: <GT> \nCandidate answer: <answer> \n\
Is the semantic meaning of the ground-truth and candidate answers similar? Answer in one word - Yes or No.'


def get_spiqa_eval_client(model_name: str = DEFAULT_SPIQA_LLM_MODEL) -> SpiQAEvalClient:
    if _SPIQA_EVAL_CLIENT.get(model_name, None) is None:
        _SPIQA_EVAL_CLIENT[model_name] = SpiQAEvalClient(model_name=model_name)
    return _SPIQA_EVAL_CLIENT[model_name]


def eval_spiqa(
    pred: Any,
    reference_answer: str = '',
    question: str = '',
    task_type: str = 'taska',
    is_boolean: bool = False,
    model_name: str = DEFAULT_SPIQA_LLM_MODEL,
    temperature: float = DEFAULT_SPIQA_TEMPERATURE,
) -> float:
    if is_boolean:
        return eval_bool_exact_match(pred, reference_answer)

    client: SpiQAEvalClient = get_spiqa_eval_client(model_name=model_name)
    gt = reference_answer if reference_answer is not None else ''

    try:
        prompt_current = _PROMPT.replace('<question>', question).replace('<GT>', gt).replace('<answer>', str(pred))
        response, prob_yes = client.call_openai_with_score(
            prompt=prompt_current,
            suffixes=_SUFFIXES_TO_SCORE,
            complement_suffixes=_COMPLEMENT_SUFFIXES,
            temperature=temperature,
            output_prefix=''
        )
        score = round(float(prob_yes), 2)
    except:
        score = 0
        print('[ERROR]: Unexpected error occurred during the evaluation of SPIQA dataset.')
    return score
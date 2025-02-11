#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import RetrieveFromDatabase, RetrieveFromVectorstore, Action, Observation
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class TwoStageHybridRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'two_stage_hybrid', max_turn: int = 2) -> None:
        super(TwoStageHybridRAGAgent, self).__init__(model, env, agent_method, max_turn)


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 database_prompt: str,
                 vectorstore_prompt: str,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 with_vision: bool = True,
                 **kwargs
    ) -> str:
        question, answer_format, pdf_context, image_message = formulate_input(dataset, example, with_vision=with_vision)
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        # 1. Generate Action
        action_prompt = f"Action 1:\n{RetrieveFromVectorstore.specification(action_format='json')}\nAction 2:\n{RetrieveFromDatabase.specification()}\n"
        prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method],
            action_prompt = action_prompt,
            question = question,
            pdf_context = pdf_context,
            database_schema = database_prompt,
            vectorstore_schema = vectorstore_prompt
        ) # system prompt + task prompt + cot thought hints
        logger.info('[Stage 1]: Generate Action ...')
        messages = [{'role': 'user', 'content': prompt}]
        if image_message:
            messages.append(image_message)
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        _, action = Action.parse_action(response, action_types=[RetrieveFromVectorstore, RetrieveFromDatabase], action_format='json', agent_method='code_block')
        logger.info(f'[Action]: {repr(action)}')
        
        # 2. Answer question
        observation: Observation = action.execute(self.env)
        if isinstance(action, RetrieveFromDatabase):
            sql = action.sql
            prompt = AGENT_PROMPTS['two_stage_text2sql'][1].format(
                system_prompt = SYSTEM_PROMPTS['two_stage_text2sql'][1],
                question = question,
                pdf_context = pdf_context,
                sql = sql,
                context = observation.obs_content,
                answer_format = answer_format
            ) # system prompt (without schema) + task prompt (insert SQL, observation) + cot thought hints
        else:
            prompt = AGENT_PROMPTS['two_stage_text2vec'][1].format(
                system_prompt = SYSTEM_PROMPTS['two_stage_text2vec'][1],
                question = question,
                pdf_context = pdf_context,
                context = observation.obs_content,
                answer_format = answer_format
            ) # system prompt + task prompt (insert retrived observation) + cot thought hints
        logger.info(f'[Stage 2]: Generate Answer ...')
        messages = [{'role': 'user', 'content': prompt}]
        if image_message:
            messages.append(image_message)
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        matched_list = re.findall(r"```(txt)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL)
        if not matched_list:
            answer = response.strip()
        else:
            answer = matched_list[-1][1].strip()
        logger.info(f'[Answer]: {answer}')
        
        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Info]: LLM API call costs ${cost:.6f}.')
        return answer
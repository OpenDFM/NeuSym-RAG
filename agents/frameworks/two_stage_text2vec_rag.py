#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import RetrieveFromVectorstore, Action, Observation
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class TwoStageText2VecRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'two_stage_text2vec', max_turn: int = 2) -> None:
        super(TwoStageText2VecRAGAgent, self).__init__(model, env, agent_method, max_turn)


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
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

        # 1. Generate RetriveFromVectorstore action
        action_prompt = f"Your output should follow the action format below:\n{RetrieveFromVectorstore.specification(action_format='json')}"
        prompt = AGENT_PROMPTS[self.agent_method][0].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method][0],
            action_prompt = action_prompt,
            question = question,
            pdf_context = pdf_context,
            vectorstore_schema = vectorstore_prompt
        ) # system prompt + action_prompt + task prompt + cot thought hints
        logger.info('[Stage 1]: Generate RetriveFromVectorstore action ...')
        messages = [{'role': 'user', 'content': prompt}]
        if image_message:
            messages.append(image_message)
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        _, action = Action.parse_action(response, action_types=[RetrieveFromVectorstore], action_format='json', agent_method='code_block')
        logger.info(f'[Action]: {repr(action)}')

        # 2. Answer question
        observation: Observation = action.execute(self.env)
        prompt = AGENT_PROMPTS[self.agent_method][1].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method][1],
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
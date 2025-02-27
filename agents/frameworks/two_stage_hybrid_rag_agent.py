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

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'two_stage_hybrid_rag', max_turn: int = 2) -> None:
        super(TwoStageHybridRAGAgent, self).__init__(model, env, agent_method, max_turn)
        self.system_prompt = SYSTEM_PROMPTS[agent_method]
        self.agent_prompt = AGENT_PROMPTS[agent_method]
        logger.info(f'[System Prompt]: stage 1 -> {self.system_prompt[0]}')
        logger.info(f'[System Prompt]: stage 2 -> {self.system_prompt[1]}')
        logger.info(f'[Agent Prompt]: stage 1 -> {self.agent_prompt[0]}')
        logger.info(f'[Agent Prompt]: stage 2 -> {self.agent_prompt[1]}')


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 database_prompt: str,
                 vectorstore_prompt: str,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 output_kwargs: Dict[str, Any] = {},
                 **kwargs
    ) -> str:
        self.env.reset()
        prev_cost = self.model.get_cost()

        # [Stage 1]: Generate RetriveFromVectorstore / RetrieveFromDatabase action
        logger.info('[Stage 1]: Generate RetriveFromVectorstore / RetrieveFromDatabase action ...')
        task_input, image_messages = formulate_input(dataset, example, use_pdf_id=True)
        logger.info(f'[Task Input]: stage 1 -> {task_input}')
        task_input = "\n".join([
            task_input,
            f"[Database Schema]: {database_prompt}",
            f"[Vectorstore Schema]: {vectorstore_prompt}"
        ])
        action_space_prompt = Action.get_action_space_prompt(
            action_types=[RetrieveFromDatabase, RetrieveFromVectorstore],
            action_format=self.env.action_format,
            interact_protocol=self.env.interact_protocol
        )
        task_prompt = self.agent_prompt[0].format(
            system_prompt=self.system_prompt[0],
            action_space_prompt=action_space_prompt,
            task_input=task_input
        )
        if image_messages:
            task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
        messages = [{'role': 'user', 'content': task_prompt}]
        response = self.model.get_response(messages, model, temperature, top_p, max_tokens)
        logger.info(f'[Response]: {response}')
        _, action = Action.parse_action(
            response,
            action_types=[RetrieveFromDatabase, RetrieveFromVectorstore],
            action_format=self.env.action_format,
            interact_protocol=self.env.interact_protocol
        )
        logger.info(f'[Action]: {repr(action)}')

        # [Stage 2]: Answer question
        logger.info(f'[Stage 2]: Generate Answer ...')
        observation: Observation = action.execute(self.env, **output_kwargs)
        task_input, _ = formulate_input(dataset, example, use_pdf_id=False)
        task_prompt = self.agent_prompt[1].format(
            system_prompt=self.system_prompt[1],
            task_input=task_input,
            context=f"[Context]: {observation.obs_content}"
        )
        logger.info(f"[Task Input]: stage 2 -> {task_prompt}")
        messages = [{'role': 'user', 'content': task_prompt}]
        response = self.model.get_response(messages, model, temperature, top_p, max_tokens)
        logger.info(f'[Response]: {response}')
        _, answer = Action.extract_thought_and_action_text(response, self.env.interact_protocol)
        logger.info(f'[Answer]: {answer}')

        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Cost]: LLM API call costs ${cost:.6f}.')
        return answer
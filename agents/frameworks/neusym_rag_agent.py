#coding=utf8
import logging, sys, os
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, HINT_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()


class NeuSymRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'neusym_rag', max_turn: int = 20) -> None:
        super(NeuSymRAGAgent, self).__init__(model, env, agent_method, max_turn)

        self.agent_prompt = AGENT_PROMPTS[agent_method].format(
            system_prompt=SYSTEM_PROMPTS[agent_method],
            action_space_prompt=env.action_space_prompt,
            hint_prompt=HINT_PROMPTS[agent_method],
            max_turn=max_turn
        )
        logger.info(f'[Agent Prompt]: {self.agent_prompt}')


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 database_prompt: str,
                 vectorstore_prompt: str,
                 window_size: int = 5,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 output_path: Optional[str] = None,
                 output_kwargs: Dict[str, Any] = {},
                 **kwargs
    ) -> str:
        # construct the initial prompt messages
        task_prompt, image_messages = formulate_input(dataset, example, use_pdf_id=True)
        logger.info(f'[Task Input]: {task_prompt}')
        # logger.info(f'[Database Schema]:\n{database_prompt}')
        # logger.info(f'[Vectorstore Schema]:\n{vectorstore_prompt}')

        task_prompt = "\n".join([
            task_prompt,
            f"[Database Schema]: {database_prompt}",
            f"[Vectorstore Schema]: {vectorstore_prompt}"
        ])
        if image_messages:
            task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
        messages = [
            {'role': 'system', 'content': self.agent_prompt},
            {'role': 'user', 'content': task_prompt}
        ]

        return self.forward(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            window_size=window_size,
            output_path=output_path,
            output_kwargs=output_kwargs
        )
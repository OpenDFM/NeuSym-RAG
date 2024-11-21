#coding=utf8
import logging, sys, os
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import Action, Observation
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, HINT_PROMPTS, AGENT_PROMPTS
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class HybridRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'react', max_turn: int = 15) -> None:
        super(HybridRAGAgent, self).__init__(model, env, agent_method, max_turn)

        self.agent_prompt = AGENT_PROMPTS[agent_method].format(
            system_prompt=SYSTEM_PROMPTS['hybrid_rag'],
            action_space_prompt=env.action_space_prompt,
            max_turn=max_turn,
            hint_prompt=HINT_PROMPTS['hybrid_rag']
        )
        logger.info(f'[AgentPrompt]: {self.agent_prompt}')


    def interact(self,
                 question: str,
                 database_prompt: str,
                 vectorstore_prompt: str,
                 answer_format: str,
                 window_size: int = 3,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 output_path: Optional[str] = None,
                 output_kwargs: Dict[str, Any] = {}
    ) -> str:
        # construct the initial prompt messages
        task_prompt = f'[Question]: {question}\n[Answer Format]: {answer_format}\n[Database Schema]: {database_prompt}\n[Vectorstore Schema]:\n{vectorstore_prompt}'
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        # logger.info(f'[Database Schema]:\n{database_prompt}')
        # logger.info(f'[Vectorstore Schema]:\n{vectorstore_prompt}')
        messages = [
            {'role': 'system', 'content': self.agent_prompt},
            {'role': 'user', 'content': task_prompt}
        ]
        answer = self.forward(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            window_size=window_size,
            output_path=output_path,
            output_kwargs=output_kwargs
        )
        return answer
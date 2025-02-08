#coding=utf8
import logging, sys, os
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import Action, Observation
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, HINT_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class Text2VecRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'react', max_turn: int = 10) -> None:
        super(Text2VecRAGAgent, self).__init__(model, env, agent_method, max_turn)

        self.agent_prompt = AGENT_PROMPTS[agent_method].format(
            system_prompt=SYSTEM_PROMPTS['text2vec'],
            action_space_prompt=env.action_space_prompt,
            max_turn=max_turn,
            hint_prompt=HINT_PROMPTS['text2vec']
        )
        logger.info(f'[AgentPrompt]: {self.agent_prompt}')


    def interact(self,
                 dataset: str, 
                 example: Dict[str, Any],
                 vectorstore_prompt: str,
                 window_size: int = 3,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 output_path: Optional[str] = None,
                 output_kwargs: Dict[str, Any] = {},
                 with_vision: bool = True,
    ) -> str:
        # construct the initial prompt messages
        question, answer_format, pdf_context, image_message = formulate_input(dataset, example, with_vision=with_vision)
        task_prompt = f'[Question]: {question}\n[Answer Format]: {answer_format}\n{pdf_context}[Vectorstore Schema]:\n{vectorstore_prompt}'
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        # logger.info(f'[Vectorstore Schema]:\n{vectorstore_prompt}')
        messages = [
            {'role': 'system', 'content': self.agent_prompt},
            {'role': 'user', 'content': task_prompt}
        ]
        if image_message:
            messages.append(image_message)
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
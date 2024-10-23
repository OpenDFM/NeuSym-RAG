#coding=utf8
import logging, sys, os
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.frameworks.agent_base import AgentBase


logging.basicConfig(encoding='utf-8')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Text2SQLRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'react', max_turn: int = 10) -> None:
        super(Text2SQLRAGAgent, self).__init__(model, env, agent_method, max_turn)

        self.agent_prompt = AGENT_PROMPTS[agent_method].format(
            system_prompt=SYSTEM_PROMPTS['text2sql'],
            action_space_prompt=env.action_space_prompt,
            max_turn=max_turn
        )


    def close(self):
        self.env.close()
        self.model.close()


    def interact(self,
                 question: str,
                 database_prompt: str,
                 answer_format: str,
                 window_size: int = 3,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 1500
    ) -> str:
        task_prompt = f'[Question]: {question}\n[Answer Format]: {answer_format}\n[Database Schema]:\n{database_prompt}'
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        messages = [
            {'role': 'system', 'content': self.agent_prompt},
            {'role': 'user', 'content': task_prompt}
        ]
        prev_cost = self.model.get_cost()
        self.env.reset()

        for turn in range(self.max_turn):
            if len(messages) > (window_size + 1) * 2: # each turn has two messages from assistant and user, respectively
                current_messages = messages[:2] + messages[-window_size * 2:]
            else: current_messages = messages
            logger.info(f'[Interaction Turn]: {turn + 1}')

            response = self.model.get_response(current_messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            logger.info(f'[Response]: {response}')

            obs, reward, flag, info = self.env.step(response)
            if info.get('parse_error', True):
                action_str = f"[Action]: failed to parse action from \"{response}\""
                logger.info(f'[ActionError]: failed to parse the Action from LLM response.')
            else:
                action = self.env.parsed_actions[-1]
                action_str = action.serialize(self.env.action_format)
                logger.info(action_str)

            observation_str = f'[Observation]:\n{obs}' if '\n' in obs else f'[Observation]: {obs}'
            logger.info(observation_str)

            if flag: # whether task is completed
                cost = self.model.get_cost() - prev_cost
                logger.info(f'[Info]: early stop at interaction turn {turn}, cost ${cost:.6f}.')
                break

            # update history messages
            messages.append({'role': 'assistant', 'content': action_str})
            messages.append({'role': 'user', 'content': observation_str})
        else:
            cost = self.model.get_cost() - prev_cost
            logger.info(f'[Warning]: exceeds the maximum interaction turn {self.max_turn}, cost ${cost:.6f}.')
        return obs
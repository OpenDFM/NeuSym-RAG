#coding=utf8
import logging, sys, os
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS
from agents.parsers import OUTPUT_PARSERS


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


class Text2SQLRAGAgent():

    def __init__(self, model: LLMClient, env: AgentEnv, method: str = 'text2sql+react', max_turns: int = 10) -> None:
        self.model = model
        self.env = env
        self.system_prompt = SYSTEM_PROMPTS[method]
        self.output_parser = OUTPUT_PARSERS[method]()
        self.max_turns = max_turns


    def get_history_messages(self, action: Dict[str, Any], observation: str) -> List[Dict[str, str]]:
        action_msg = {
            "role": "assistant",
            "content": self.env.serialize_action(action)
        }
        obs_msg = {
            'role': "user",
            "content": f'Observation:\n{observation}' if '\n' in observation else f'Observation: {observation}'
        }
        return [action_msg, obs_msg]


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
        task_prompt = f'Question: {question}\nAnswer Format: {answer_format}\nDatabase Schema:\n{database_prompt}'
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': task_prompt}
        ]
        prev_cost = self.model.get_cost()
        self.env.reset()
        
        for turn in range(self.max_turns):
            if len(messages) > (window_size + 1) * 2: # each turn has two messages from assistant and user, respectively
                current_messages = messages[:2] + messages[-window_size * 2:]
            else: current_messages = messages
            logger.info(f'[Interaction Turn]: {turn + 1}')

            response = self.model.get_response(current_messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            logger.info(f'[Response]: {response}')

            action = self.output_parser.parse(response)
            logger.info(f'[Action]: {action}')

            obs, reward, flag, info = self.env.step(action)
            logger.info(f'[Observation]:\n{obs}' if '\n' in obs else f'[Observation]: {obs}')
            # update history messages
            if flag:
                cost = self.model.get_cost() - prev_cost
                logger.info(f'[Info]: early stop at interaction turn {turn}, cost ${cost:.6f}.')
                break
            messages.extend(self.get_history_messages(action, obs))
        else:
            cost = self.model.get_cost() - prev_cost
            logger.info(f'[Warning]: exceeds the maximum interaction turns {self.max_turns}, cost ${cost:.6f}.')
        return obs
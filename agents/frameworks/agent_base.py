#coding=utf8
import logging, json, tiktoken
from tiktoken import Encoding
from abc import ABC, abstractmethod
from agents.envs import AgentEnv
from agents.envs.actions import Action, Observation
from agents.models import LLMClient
from typing import List, Dict, Any, Union, Tuple, Optional


logger = logging.getLogger()

class AgentBase(ABC):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'react', max_turn: int = 10):
        self.model, self.env = model, env
        self.agent_method, self.max_turn = agent_method, max_turn
        self.agent_prompt = ''
        self.encoding_models = dict()


    def close(self):
        self.env.close()
        self.model.close()


    @abstractmethod
    def interact(self, *args, **kwargs) -> str:
        pass


    def truncate_tokens(self, text: str, max_tokens: int = 30, encoding_model: str = 'cl100k_base') -> str:
        """ Given a text string, truncate it to max_tokens using encoding_model tokenizer
        """
        if encoding_model not in self.encoding_models:
            encoding: Encoding = tiktoken.get_encoding(encoding_model)
            self.encoding_models[encoding_model] = encoding
        encoding: Encoding = self.encoding_models[encoding_model]
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens * 1000:
            tokens = tokens[:max_tokens * 1000]
            text = encoding.decode(tokens)
        return text


    def forward(self, messages: List[Dict[str, Any]], model: str = '', temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 1500, window_size: int = 3, output_path: Optional[str] = None, output_kwargs: Dict[str, Any] = {}) -> str:
        prev_cost = self.model.get_cost()
        self.env.reset()

        for turn in range(self.max_turn):
            if len(messages) > (window_size + 1) * 2: # each turn has two messages from assistant and user, respectively
                current_messages = messages[:2] + messages[-window_size * 2:]
            else: current_messages = messages
            logger.info(f'[Interaction Turn]: {turn + 1}')

            response = self.model.get_response(current_messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            logger.debug(f'[Response]: {response}')

            obs, reward, flag, info = self.env.step(response, **output_kwargs)
            action: Action = self.env.parsed_actions[-1]
            action_msg = action.convert_to_message(self.env.action_format)
            logger.info(action_msg['content'])

            obs: Observation
            obs_msg = obs.convert_to_message()
            if isinstance(obs_msg['content'], list): # array of messages, see doc: https://platform.openai.com/docs/guides/vision#uploading-base64-encoded-images
                for obs_msg_content_item in obs_msg['content']:
                    if obs_msg_content_item['type'] == 'text':
                        logger.info(obs_msg_content_item['text'])
            else:
                logger.info(obs_msg['content'])

            # update history messages
            messages.append(action_msg)
            messages.append(obs_msg)

            if flag: # whether task is completed
                cost = self.model.get_cost() - prev_cost
                logger.info(f'[Info]: early stop at interaction turn {turn + 1}, cost ${cost:.6f}.')
                break
        else:
            cost = self.model.get_cost() - prev_cost
            logger.info(f'[Warning]: exceeds the maximum interaction turn {self.max_turn}, cost ${cost:.6f}.')
        if output_path is not None:
            with open(output_path, 'w', encoding='utf-8') as f:
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False) + '\n')
        return obs.obs_content
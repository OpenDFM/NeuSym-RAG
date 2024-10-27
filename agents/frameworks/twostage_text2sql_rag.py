#coding=utf8
import logging, sys, os
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import GenerateSQL, GenerateAnswer
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class TwoStageText2SQLRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'twostage', max_turn: int = 2) -> None:
        super(TwoStageText2SQLRAGAgent, self).__init__(model, env, agent_method, max_turn)
        self.action_order: List[type] = [GenerateSQL, GenerateAnswer]
        self.agent_prompt = [
            AGENT_PROMPTS[agent_method][turn].format(
                system_prompt=SYSTEM_PROMPTS['twostage_text2sql'][turn]
            )
            for turn in range(max_turn)
        ]
        logger.info(f'[AgentPrompt_0]: {self.agent_prompt[0]}')
        logger.info(f'[AgentPrompt_1]: {self.agent_prompt[1]}')

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
        task_prompt = f'[Question]: {question}\n[Answer Format]: {answer_format}\n'
        database_prompt = f'[Database Schema]:\n{database_prompt}\n'
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        messages = []
        prev_cost = self.model.get_cost()
        self.env.reset()


        # 1. Generate SQL
        prompt = ...   # system prompt + task prompt + cot thought hints
        response = self.model.get_response()
        sql = ... # parse SQL from response

        # 2. Answer question
        from envs.actions import GenerateSQL
        action = GenerateSQL(sql='xxx', observation_format_kwargs={...})
        observation = GenerateSQL.execute(action, self.env)
        prompt = ... # system prompt + task prompt (insert SQL, observation) + cot thought hints
        response = self.model.get_response()
        answer = ... # extract answer from response
        return answer

        for turn in range(self.max_turn):
            if turn == 0:
                current_messages = [
                    {'role': 'system', 'content': self.agent_prompt[turn]},
                    {'role': 'user', 'content': task_prompt + database_prompt}
                ]
            else:
                current_messages = [
                    {'role': 'system', 'content': self.agent_prompt[turn]},
                    {'role': 'user', 'content': task_prompt}
                ] + messages
            logger.info(f'[Interaction Turn]: {turn + 1}')
            logger.info(f'[Messages]: {current_messages}')

            response = self.model.get_response(current_messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            logger.info(f'[Response]: {response}')

            obs, reward, flag, info = self.env.step(response)
            if info.get('parse_error', True):
                action_str = f"[Action]: failed to parse action from \"{response}\""
                logger.info(f'[ActionError]: failed to parse the Action from LLM response.')
            else:
                action = self.env.parsed_actions[-1]
                if not isinstance(action, self.action_order[turn]):
                    logger.info(f'[ActionError]: performed wrong action on turn {turn + 1}')
                action_str = action.serialize(self.env.action_format)
                logger.info(action_str)

            observation_str = f'[Observation]:\n{obs}' if '\n' in str(obs) else f'[Observation]: {obs}'
            logger.info(observation_str)

            if flag: # whether task is completed
                cost = self.model.get_cost() - prev_cost
                logger.info(f'[Info]: early stop at interaction turn {turn + 1}, cost ${cost:.6f}.')
                break

            # update history messages
            messages.append({'role': 'assistant', 'content': action_str})
            messages.append({'role': 'user', 'content': observation_str})
        else:
            cost = self.model.get_cost() - prev_cost
            logger.info(f'[Warning]: exceeds the maximum interaction turn {self.max_turn}, cost ${cost:.6f}.')
        return obs
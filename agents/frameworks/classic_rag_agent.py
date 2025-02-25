#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks import AgentBase
from agents.envs.actions import ClassicRetrieve, Action, Observation


logger = logging.getLogger()


class ClassicRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'classic_rag', max_turn: int = 1) -> None:
        super(ClassicRAGAgent, self).__init__(model, env, agent_method, max_turn)

        self.system_prompt: str = SYSTEM_PROMPTS[agent_method]
        self.agent_prompt: str = AGENT_PROMPTS[agent_method]
        logger.info(f"[System Prompt]: {self.system_prompt}")
        logger.info(f"[Agent Prompt]: {self.agent_prompt}")


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 table_name: Optional[str] = None,
                 column_name: Optional[str] = None,
                 collection_name: str = 'text_sentence_transformers_all_minilm_l6_v2',
                 limit: int = 4,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 **kwargs
    ) -> str:
        self.env.reset()

        # [Stage 1]: Retrieve the context (via hard coding)
        filter_condition = ""
        if len(example['anchor_pdf']) == 1:
            pdf_string = repr(example['anchor_pdf'][0])
            filter_condition = f"pdf_id == '{pdf_string}'"
        elif len(example['anchor_pdf']) > 1:
            pdf_string = ', '.join([repr(pid) for pid in example['anchor_pdf']])
            filter_condition = f"pdf_id in [{pdf_string}]"
        assert collection_name in self.env.vectorstore_conn.list_collections(), f'Collection {collection_name} not found in the vectorstore connection.'
        ClassicRetrieve.set_default(
            collection_name=collection_name,
            table_name=table_name,
            column_name=column_name,
            filter=filter_condition
        )
        action = ClassicRetrieve(query=example['question'], limit=limit)
        observation: Observation = action.execute(self.env)
        logger.info(f'[Stage 1]: Retrieve top {limit} context from {collection_name} ...')
        logger.info(f'[Context]: {observation.obs_content}')

        # [Stage 2]: Answer the question
        logger.info('[Stage 2]: Generate Answer ...')
        prev_cost = self.model.get_cost()
        task_input, image_messages = formulate_input(dataset, example, use_pdf_id=False)
        task_prompt = self.agent_prompt.format(
            system_prompt=self.system_prompt,
            task_input=task_input,
            context=f"[Context]: {observation.obs_content}"
        )
        logger.info(f'[Stage 2]: Task Prompt:\n{task_prompt}')
        if image_messages:
            task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
        messages = [
            {'role': 'user', 'content': task_prompt}
        ]
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        _, answer = Action.extract_thought_and_action_text(response, interact_protocol=self.env.interact_protocol)
        logger.info(f'[Response]: {response}')
        logger.info(f'[Answer]: {answer}')

        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Cost]: LLM API call costs ${cost:.6f}.')
        return answer
#coding=utf8
import logging, subprocess, json, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv, Action
from agents.models import LLMClient
from agents.prompts import AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()


class TwoStageGraphRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'two_stage_graph_rag', max_turn: int = 2) -> None:
        super(TwoStageGraphRAGAgent, self).__init__(model, env, agent_method, max_turn)
        self.agent_prompt = AGENT_PROMPTS[agent_method]
        logger.info(f'[Agent Prompt]: {self.agent_prompt}')


    def get_graphrag_response(self, graphrag_query: str, graphrag_root: str, graphrag_method: str) -> str:
        command = [
            'graphrag', 'query',
            '--root', graphrag_root,
            '--method', graphrag_method,
            '--query', graphrag_query
        ]
        process = subprocess.run(command, text=True, capture_output=True)
        response = process.stdout
        logger.info(f'[Response]: {response}')
        _, answer = Action.extract_thought_and_action_text(response, self.env.interact_protocol)
        logger.info(f'[Answer]: {answer}')
        return answer


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 graphrag_root: str = 'data/graph',
                 graphrag_method: str = 'local',
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 output_path: Optional[str] = None,
                 **kwargs
    ) -> str:
        self.env.reset()

        assert graphrag_method in ['local', 'global'], f"Invalid GraphRAG method: {graphrag_method}"

        # [Stage 1]: construct the `graphrag_query`
        task_input, _ = formulate_input(dataset, example, use_pdf_id=False)
        graphrag_query = self.agent_prompt.format(task_input=task_input)

        # [Stage 2]: retrieve and generate the answer via official script
        logger.info(f"[Task Input]: {graphrag_query}")
        answer = self.get_graphrag_response(graphrag_query, graphrag_root, graphrag_method)

        messages = [
            {"role": "user", "content": graphrag_query},
            {"role": "assistant", "content": answer}
        ]
        if output_path is not None:
            with open(output_path, 'w', encoding='utf-8') as of:
                for msg in messages:
                    of.write(json.dumps(msg, ensure_ascii=False) + '\n')
        return answer
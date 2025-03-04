#coding=utf8
import logging, json, sys, os, subprocess, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv, Action
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()


class IterativeGraphRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'iterative_graph_rag', max_turn: int = 10) -> None:
        super(IterativeGraphRAGAgent, self).__init__(model, env, agent_method, max_turn)
        self.system_prompt = SYSTEM_PROMPTS[agent_method]
        self.agent_prompt = AGENT_PROMPTS[agent_method]
        logger.info(f'[System Prompt]: {self.system_prompt}')
        logger.info(f'[Agent Prompt]: graph query -> {self.agent_prompt[0]}')
        logger.info(f'[Agent Prompt]: answer judgement -> {self.agent_prompt[1]}')


    def replace_question_in_task_input(self, task_input: str, question: str) -> str:
        return re.sub(r"(\[Question\]: )(.*?)(?=$|\[Conference|\[Anchor PDF|\[Reference PDF)", lambda m: m.group(1) + question + '\n', task_input, flags=re.DOTALL)


    def get_graphrag_output(self, graphrag_query: str, graphrag_root: str, graphrag_method: str) -> Tuple[str, str]:
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
        return response, answer


    def interact(self,
                 dataset: str, 
                 example: Dict[str, Any],
                 graphrag_root: Optional[str] = None,
                 graphrag_method: str = 'local',
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 output_path: Optional[str] = None,
                 **kwargs
    ) -> str:
        self.env.reset()
        prev_cost = self.model.get_cost()

        assert graphrag_method in ['local', 'global'], f"Invalid GraphRAG method: {graphrag_method}"

        # construct the initial task prompt
        initial_task_input, _ = formulate_input(dataset, example, use_pdf_id=True)
        logger.info(f'[Task Input]: {initial_task_input}')
        graphrag_query = self.agent_prompt[0].format(task_input=initial_task_input)
        response, initial_answer = self.get_graphrag_output(graphrag_query, graphrag_root, graphrag_method)

        turn, task_input, answer = 0, initial_task_input, initial_answer
        messages = [
            {"role": "user", "content": graphrag_query},
            {"role": "assistant", "content": response}
        ]
        while turn < self.max_turn:
            # get model judgement
            task_prompt = self.agent_prompt[1].format(
                system_prompt=self.system_prompt,
                task_input=initial_task_input,
                response=response
            )
            current_messages = [{"role": "user", "content": task_prompt}]
            messages.extend(current_messages)
            logger.info(f"[Turn {turn + 1}]: {task_prompt}")
            response = self.model.get_response(current_messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            _, judgement = Action.extract_thought_and_action_text(response, self.env.interact_protocol)
            logger.info(f'[Response]: {response}')
            logger.info(f"[Judgement]: {judgement}")
            messages.append({"role": "assistant", "content": response})

            if judgement.strip().lower() == "completed":
                # the model thinks the answer is completed
                messages.append({"role": "assistant", "content": answer})
                break

            turn += 1
            if turn >= self.max_turn:
                # fallback to the trivial Two-stage Graph-RAG method
                answer = initial_answer
                messages.append({"role": "assistant", "content": answer})
                break

            task_input = self.replace_question_in_task_input(initial_task_input, judgement.strip())
            graphrag_query = self.agent_prompt[0].format(task_input=task_input)
            response, answer = self.get_graphrag_output(graphrag_query, graphrag_root, graphrag_method)
            messages.append({"role": "user", "content": graphrag_query})
            messages.append({"role": "assistant", "content": response})

        if output_path is not None:
            with open(output_path, 'w', encoding='utf8') as of:
                for msg in messages:
                    of.write(json.dumps(messages, ensure_ascii=False) + '\n')
        
        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Cost]: LLM API call costs ${cost:.6f}.')
        return answer
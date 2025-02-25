#coding=utf8
import logging, re, os, json
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv, Action
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS, formulate_input
from agents.frameworks import AgentBase
from utils.airqa_utils import get_airqa_paper_metadata
from utils.dataset_utils import DATASET_DIR
from utils.functions import get_pdf_page_text, truncate_tokens


logger = logging.getLogger()


class TrivialBaselineAgent(AgentBase):

    def __init__(self,
                 model: LLMClient,
                 env: AgentEnv,
                 agent_method: str = 'trivial_question_only',
                 max_turn: int = 1
        ) -> None:
        assert agent_method in ['trivial_question_only', 'trivial_title_with_abstract', 'trivial_full_text_with_cutoff'], f'Invalid agent method: {agent_method} for {self.__class__.__name__}.'
        super(TrivialBaselineAgent, self).__init__(model, env, agent_method, max_turn)
        self.system_prompt = SYSTEM_PROMPTS[agent_method]
        self.agent_prompt = AGENT_PROMPTS[agent_method]
        logger.info(f"[System Prompt]: {self.system_prompt}")
        logger.info(f"[Agent Prompt]: {self.agent_prompt}")


    def get_pdf_context(self, pid: str, uuid2papers: Dict[str, Dict[str, Any]]) -> str:
        if self.agent_method == 'trivial_title_with_abstract':
            meta = uuid2papers[pid]
            return f"[Title]: {meta['title']}\n[Abstract]: {meta['abstract']}"
        else:
            ppath = os.path.join('data', 'dataset', self.env.dataset, 'processed_data', f'{pid}.json')
            if os.path.exists(ppath):
                with open(ppath, 'r', encoding='utf-8') as fin:
                    pdata = json.load(fin)
                return '\n'.join(toc['title'] + '\n' + toc['text'] for toc in pdata['info_from_mineru']['TOC'])
            else:
                pdf_path = uuid2papers[pid]['pdf_path']
                contents = get_pdf_page_text(pdf_path, generate_uuid=False)['page_contents']
                return "\n\n".join(contents)


    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 cutoff: int = 5,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 **kwargs
    ) -> str:
        self.env.reset()
        
        # [Stage 0]: Prepare the task input and optional context
        logger.info('[Stage 0]: Extract context ...')
        task_input, image_messages = formulate_input(dataset, example, use_pdf_id=False)
        if self.agent_method == 'trivial_question_only':
            task_input = f"[Question]: {example['question']}\n[Answer Format]: {example['answer_format']}"
            context = 'No context provided.'
        else:
            dataset_dir = os.path.join(DATASET_DIR, dataset)
            uuid2papers = get_airqa_paper_metadata(dataset_dir=dataset_dir)
            context_list = [self.get_pdf_context(pid, uuid2papers) for pid in example["anchor_pdf"]]
            if self.agent_method == 'trivial_title_with_abstract':
                pdf_separator = '\n\n'
                context = 'Titles and abstracts for anchor PDFs:\n' + pdf_separator.join(context_list)
            else: # trivial_full_text_with_cutoff
                pdf_separator = '\n\n' + '-' * 10 + '\n\n'
                context = f'Full text for anchor PDFs with {cutoff}k tokens cutoff:\n' + pdf_separator.join(context_list)
            context = truncate_tokens(context, max_tokens=cutoff)
        task_prompt = self.agent_prompt.format(
            task_input=task_input,
            context=f"[Context]: {context}"
        )
        logger.info(f"[Task Input]: {task_input}")
        logger.info(f'[Context]: {context}')
        if image_messages:
            task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
        messages = [{'role': 'user', 'content': task_prompt}]

        # [Stage 1]: Answer the user question
        logger.info('[Stage 1]: Generate Answer ...')
        prev_cost = self.model.get_cost()
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        _, answer = Action.extract_thought_and_action_text(response, interact_protocol=self.env.interact_protocol)
        logger.info(f'[Response]: {response}')
        logger.info(f'[Answer]: {answer}')
        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Cost]: LLM API call costs ${cost:.6f}.')
        return answer
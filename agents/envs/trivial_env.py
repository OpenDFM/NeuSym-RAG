#coding=utf8
import json, os, re
from agents.envs.env_base import AgentEnv
from typing import Optional

class TrivialEnv(AgentEnv):
    """ Responsible for managing the environment for the trivial retrieval, which includes getting text contents from PDF files.
    """

    def __init__(self, agent_method: Optional[str] = 'trivial', dataset: Optional[str] = None, **kwargs) -> None:
        super(TrivialEnv, self).__init__(agent_method=agent_method, dataset=dataset)
        self.pdf_contents = dict()
        if dataset in ['pdfvqa', 'tatdqa']:
            with open(os.path.join('data', 'dataset', dataset, 'processed_data', 'pdf_data.jsonl'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    pdf_data = json.loads(line)
                    pdf_id = pdf_data['pdf_id']
                    self.pdf_contents[pdf_id] = dict()
                    for page_info in pdf_data['page_infos']:
                        page_number = page_info['page_number']
                        page_content = re.sub(r'\n+', '\n', '\n'.join(text.strip() for text in page_info['bbox_text']))
                        if page_content != '':
                            self.pdf_contents[pdf_id][page_number] = page_content
        else:
            raise ValueError(f"Dataset {dataset} is not supported.")

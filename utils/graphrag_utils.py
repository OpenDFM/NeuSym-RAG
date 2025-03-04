#coding=utf8
import argparse, json, os, yaml, sys, re, subprocess
from typing import Any, Dict, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions import get_pdf_page_text
from utils.config import DATASET_DIR, get_graphrag_root
from utils.airqa_utils import get_airqa_paper_metadata
from agents.models import infer_model_class, GPTClient, VLLMClient


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as fin:
        data = yaml.safe_load(fin)
    return data


def write_yaml(data: dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as fout:
        yaml.safe_dump(data, fout)
    return


def config_graphrag_settings(dataset: str, llm: str, embed_model: str):
    """ Overwrite the default settings.yaml and .env files in the graphrag root directory.
    Remember to set the environment variable `OPENAI_API_KEY` and `OPENAI_BASE_URL` for OpenAI API, or
        `VLLM_API_KEY`, `VLLM_BASE_URL` and `VLLM_EMBED_BASE_URL` for vLLM API.
    """
    graphrag_root = get_graphrag_root(dataset)
    env_path = os.path.join(graphrag_root, '.env')
    settings_path = os.path.join(graphrag_root, 'settings.yaml')
    settings = load_yaml(settings_path)
    settings['llm']['model'] = llm
    settings['embeddings']['llm']['model'] = embed_model # e.g.,'text-embedding-3-small'

    # configure settings.yaml
    model_cls = infer_model_class(llm)
    if isinstance(model_cls, GPTClient):
        openai_api_base = os.environ.get('OPENAI_BASE_URL', "https://api.openai.com/v1").rstrip('/')
        api_key = os.environ['OPENAI_API_KEY']
        # summary llm
        settings['llm']['api_base'] = openai_api_base
        # embedding model
        settings['embeddings']['llm']['api_base'] = openai_api_base
    else: # VLLMClient, open-source LLM with local server
        vllm_api_base = os.environ.get('VLLM_BASE_URL', "http://localhost:8000/v1").rstrip('/')
        vllm_embed_api_base = os.environ.get('VLLM_EMBED_BASE_URL', "http://localhost:8001/v1").rstrip('/')
        api_key = os.environ.get('VLLM_API_KEY', "EMPTY")
        # summary llm
        settings['llm']['api_base'] = vllm_api_base
        # embedding model
        settings['embeddings']['llm']['api_base'] = vllm_embed_api_base

    # overwrite the .env file
    with open(env_path, 'w', encoding='utf-8') as fout:
        fout.write(f"GRAPHRAG_API_KEY={api_key}\n")
    # write back to settings.yaml
    write_yaml(settings, settings_path)
    return graphrag_root


def get_pdf_text(dataset: str, pdf_id: str) -> str:
    uuid2papers = get_airqa_paper_metadata(dataset_dir=os.path.join(DATASET_DIR, dataset))
    pdf_path = uuid2papers[pdf_id]['pdf_path']
    pdf_text = get_pdf_page_text(pdf_path, generate_uuid=False, normalize_blank=False)['page_contents']
    pdf_text = "\n\n".join(pdf_text)
    return pdf_text.strip()


def get_graphrag_input(dataset: str, test_data: str):
    graphrag_root = get_graphrag_root(dataset)
    graphrag_input_dir = os.path.join(graphrag_root, 'input')
    os.makedirs(graphrag_input_dir, exist_ok=True)
    test_data_path = os.path.join(DATASET_DIR, dataset, test_data) if not os.path.exists(test_data) else test_data
    with open(test_data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            example = json.loads(line)
            for pdf_id in example['anchor_pdf'] + example['reference_pdf']:
                fout_name = os.path.join(graphrag_input_dir, pdf_id + '.txt')
                if not os.path.exists(fout_name):
                    pdf_text = get_pdf_text(dataset, pdf_id)
                    with open(fout_name, 'w', encoding='utf-8') as fout:
                        fout.write(pdf_text)
    return graphrag_root


def init_graph(dataset: str):
    graphrag_root = get_graphrag_root(dataset)
    subprocess.run([
        'graphrag', 'init', '--root', graphrag_root
    ])
    return


def build_graph(dataset: str, llm: str, embed_model: str):
    graphrag_root = config_graphrag_settings(dataset, llm, embed_model)
    # graphrag_output_dir = os.path.join(graphrag_root, 'output')
    # os.makedirs(graphrag_output_dir, exist_ok=True)
    subprocess.run([
        'graphrag', 'index', '--root', graphrag_root
    ])
    return


def test_query(dataset: str, test_data: str, llm: str, embed_model: str, graphrag_method: str = 'local'):
    graphrag_root = config_graphrag_settings(dataset, llm, embed_model)
    test_data_path = os.path.join(DATASET_DIR, dataset, test_data) if not os.path.exists(test_data) else test_data
    with open(test_data_path, 'r', encoding='utf8') as inf:
        for line in inf:
            if line.strip() == '': continue
            question = json.loads(line)['question']
            print('[Question]:', question)
            command = [
                'graphrag', 'query',
                '--root', graphrag_root,
                '--method', graphrag_method,
                '--query', question
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            response = process.stdout
            print(f'[Response]: {response}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['airqa', 'm3sciqa', 'scidqa'], help='dataset name')
    parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='LLM model name')
    parser.add_argument('--graphrag_embed', type=str, default='text-embedding-3-small', help='embedding model name')
    parser.add_argument('--graphrag_method', type=str, default='local', help='graphrag method')
    parser.add_argument('--function', type=str, default="init_graph", choices=['init_graph', 'gather_input', 'build_graph', 'test_query'], help='which function to use')
    args = parser.parse_args()

    if args.function == 'init_graph':
        init_graph(args.dataset)
    elif args.function == 'gather_input':
        get_graphrag_input(args.dataset, args.test_data)
    elif args.function == 'build_graph':
        build_graph(args.dataset, args.llm, args.graphrag_embed)
    elif args.function == 'test_query':
        test_query(args.dataset, args.test_data, args.llm, args.graphrag_embed, args.graphrag_method)
    else:
        raise ValueError('Invalid function name.')

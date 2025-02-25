#coding=utf8
import argparse, json, os, yaml, sys
from typing import Any, Dict, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions import get_pdf_page_text
from utils.airqa_utils import get_airqa_paper_metadata, AIRQA_DIR


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def write_yaml(data: dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as fout:
        yaml.dump(data, fout)
    return


def config_graphrag_settings(args: argparse.Namespace):
    settings_path = os.path.join(args.graphrag_root, 'settings.yaml')
    settings = load_yaml(settings_path)
    settings['llm']['model'] = args.llm
    settings['llm']['api_base'] = os.environ.get('OPENAI_BASE_URL', "").rstrip('/')
    settings['llm']['api_key'] = os.environ['OPENAI_API_KEY']
    write_yaml(settings, settings_path)
    return


def get_pdf_text(dataset: str, pdf_id: str) -> str:
    uuid2papers = get_airqa_paper_metadata(dataset_dir=os.path.join(os.path.dirname(AIRQA_DIR), dataset))
    pdf_path = uuid2papers[pdf_id]['pdf_path']
    pdf_text = get_pdf_page_text(pdf_path, generate_uuid=False)['page_contents']
    pdf_text = "\n\n".join(pdf_text)
    return pdf_text.strip()


def get_graphrag_input(dataset: str, test_data: str):
    graphrag_input_dir = os.path.join('graphrag', dataset, 'input')
    os.makedirs(graphrag_input_dir, exist_ok=True)
    with open(os.path.join('data', 'dataset', dataset, test_data), 'r', encoding='utf-8') as fin:
        for line in fin:
            example = json.loads(line)
            for pdf_id in example['anchor_pdf'] + example['reference_pdf']:
                fout_name = os.path.join(graphrag_input_dir, pdf_id + '.txt')
                if not os.path.exists(fout_name):
                    pdf_text = get_pdf_text(dataset, pdf_id)
                    with open(fout_name, 'w', encoding='utf-8') as fout:
                        fout.write(pdf_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='airqa', help='dataset name')
    parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
    args = parser.parse_args()
    get_graphrag_input(args.dataset, args.test_data)

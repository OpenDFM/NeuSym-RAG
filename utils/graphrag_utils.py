import argparse, json, os
from PyPDF2 import PdfReader


def get_pdf_text(dataset: str, pdf_id: str) -> str:
    with open(os.path.join('data', 'dataset', dataset, 'metadata', pdf_id + '.json'), 'r', encoding='utf-8') as fin:
        metadata = json.load(fin)
    pdf_text = ''
    with open(os.path.join('data', 'dataset', dataset, 'papers', metadata['conference'].lower() + str(metadata['year']), pdf_id + '.pdf'), 'rb') as fin:
        pdf_reader = PdfReader(fin)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
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

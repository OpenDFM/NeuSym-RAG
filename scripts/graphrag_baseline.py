#coding=utf8
import argparse, json, logging, os, subprocess, sys, yaml, re
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_utils import evaluate, print_result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='which dataset to use')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
parser.add_argument('--graphrag_root', type=str, required=True, help='root directory of graphrag')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results')
args = parser.parse_args()

with open(os.path.join(args.graphrag_root, 'settings.yaml'), 'r', encoding='utf-8') as inf:
    llm_name = yaml.safe_load(inf)['llm']['model']
filename = f'{args.dataset}_graphrag_{llm_name}'
result_dir = os.path.join(args.result_dir, filename)
os.makedirs(result_dir, exist_ok=True)

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(os.path.join(result_dir, 'log.txt'), encoding='utf-8')
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

test_data = []
if os.path.exists(args.test_data) and os.path.isfile(args.test_data):
    test_data_path = args.test_data
elif os.path.exists(os.path.join('data', 'dataset', args.dataset, args.test_data)):
    test_data_path = os.path.join('data', 'dataset', args.dataset, args.test_data)
else:
    test_data_path = os.path.join('data', 'dataset', args.dataset, 'processed_data', args.test_data)
with open(test_data_path, 'r', encoding='utf-8') as inf:
    for line in inf:
        test_data.append(json.loads(line))

start_time = datetime.now()
preds = []
for data_idx, data in enumerate(test_data):
    logger.info(f"Processing question [{data_idx + 1}/{len(test_data)}]: {data['uuid']}")
    output_path = os.path.join(result_dir, f"{data['uuid']}.txt")
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as inf:
            result = inf.read().strip()
    else:
        try:
            graphrag_query = f"[Question]: {data['question'].strip()}\n[Answer Format]: {data['answer_format'].strip()}\nYou should wrap your answer in three backticks:\n```txt\nfinal answer\n```"
            if len(data['anchor_pdf']) > 0:
                anchor_pdf_titles = []
                for anchor_pdf_id in data['anchor_pdf']:
                    with open(os.path.join('data', 'dataset', args.dataset, 'metadata', anchor_pdf_id + '.json'), 'r', encoding='utf-8') as inf:
                        anchor_pdf_titles.append(json.load(inf)['title'].replace('\n', ' ').strip())
                anchor_pdf_titles_str = '\n'.join(anchor_pdf_titles)
                graphrag_query += f"\n[Related Paper Titles]:\n{anchor_pdf_titles_str}"
            command = [
                'graphrag', 'query',
                '--root', args.graphrag_root,
                '--method', 'global' if 'retrieval' in data['tags'] else 'local',
                '--query', graphrag_query
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            print(process.stderr)
            result = process.stdout
            match = re.search(r"```txt(.*?)```", result.strip())
            if match:
                result = match.group(1).strip()
            else:
                result = ""
        except Exception as e:
            logger.error(f"[❌Error❌]: ({data['uuid']}) {str(e)}")
            result = '[ERROR]: ' + str(e)
        with open(output_path, 'w', encoding='utf-8') as ouf:
            ouf.write(result)
    preds.append({'uuid': data['uuid'], 'answer': result})

output_path = os.path.join(result_dir, 'result.jsonl')
with open(output_path, 'w', encoding='utf-8') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")

if not args.no_eval:
    result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
    result_table = print_result(result)
    logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

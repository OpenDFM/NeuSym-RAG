#coding=utf8
import argparse, json, logging, os, subprocess, sys, yaml, re
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.models import infer_model_class, LLMClient
from utils.eval_utils import evaluate, print_result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='airqa', help='which dataset to use')
parser.add_argument('--test_data', type=str, default='test_data_553.jsonl', help='test data file')
parser.add_argument('--graphrag_root', type=str, required=True, help='root directory of graphrag')
parser.add_argument('--api_key', type=str, default='OPENAI', choices=['OPENAI', 'EMPTY'], help='API key')
parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='llm name')
parser.add_argument('--llm_port', type=int, help='open-source llm port')
parser.add_argument('--emb_model', type=str, default='bge-large-en-v1.5', help='open-source embedding model name')
parser.add_argument('--emb_model_port', type=int, help='open-source embedding model port')
parser.add_argument('--max_turn', type=int, required=True, help='maximum number of turns')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results')
args = parser.parse_args()

for pdf_id in os.listdir(os.path.join(args.graphrag_root, 'graphs')):
    with open(os.path.join(args.graphrag_root, 'graphs', pdf_id, '.env'), 'w', encoding='utf-8') as ouf:
        ouf.write(f"GRAPHRAG_API_KEY={os.environ['OPENAI_API_KEY'] if args.api_key == 'OPENAI' else args.api_key}\n")
    with open(os.path.join(args.graphrag_root, 'graphs', pdf_id, 'settings.yaml'), 'r', encoding='utf-8') as inf:
        settings = yaml.safe_load(inf)
    settings['llm']['model'] = args.llm
    settings['llm']['api_base'] = os.environ['OPENAI_BASE_URL'].strip('/') if args.api_key == 'OPENAI' else f'http://localhost:{args.llm_port}/v1'
    settings['embeddings']['llm']['model'] = 'text-embedding-3-small' if args.api_key == 'OPENAI' else args.emb_model
    settings['embeddings']['llm']['api_base'] = os.environ['OPENAI_BASE_URL'].strip('/') if args.api_key == 'OPENAI' else f'http://localhost:{args.emb_model_port}/v1'
    with open(os.path.join(args.graphrag_root, 'graphs', pdf_id, 'settings.yaml'), 'w', encoding='utf-8') as ouf:
        yaml.safe_dump(settings, ouf)
filename = f'{args.dataset}_graphrag_{args.llm}'
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

llm: LLMClient = infer_model_class(args.llm)()

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
    elif len(data['anchor_pdf']) == 0:
        result = ""
    else:
        try:
            question = data['question'].strip()
            for turn in range(args.max_turn):
                command = [
                    'graphrag', 'query',
                    '--root', os.path.join(args.graphrag_root, 'graphs', data['anchor_pdf'][0]),
                    '--method', 'local',
                    '--query', f"[Question]: {question}\n[Answer Format]: {data['answer_format'].strip()}\nYou should wrap your answer in three backticks:\n```txt\nfinal answer\n```"
                ]
                process = subprocess.run(command, text=True, capture_output=True)
                print(process.stderr)
                result = process.stdout
                match = re.search(r"```txt(.*?)```", result.strip())
                if match:
                    result = match.group(1).strip()
                else:
                    result = ""
                if turn == args.max_turn - 1:
                    break
                response = llm.get_response(
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are a helpful assistant. Now you need to determine whether the answer is valid for the question. If valid, please output only 1 word "valid". If not valid, please refine the question and output it in three backticks:\n```txt\nrefined question\n```'
                        },
                        {
                            'role': 'user',
                            'content': f"[Question]: {question}\n[Answer]: {result}"
                        }
                    ],
                    model=args.llm
                )
                if response.strip().lower() == 'valid':
                    break
                match = re.search(r"```txt(.*?)```", response.strip())
                if match:
                    question = match.group(1).strip()
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

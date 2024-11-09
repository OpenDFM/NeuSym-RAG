#coding=utf8
import argparse, os, sys, json, logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.envs import ENVIRONMENTS, AgentEnv
from agents.models import infer_model_class, LLMClient
from agents.frameworks import FRAMEWORKS, AgentBase
from agents.prompts import formulate_input
from utils.eval_utils import evaluate, print_result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pdfvqa', help='which dataset to use')
parser.add_argument('--vectorstore', type=str, default='biology_paper', help='which vectorstore to use')
parser.add_argument('--test_data', type=str, default='test_data_sample.jsonl', help='test data file')
parser.add_argument('--table_name', type=str, default=None, help='which table to use, if not specified, use all tables under the database')
parser.add_argument('--column_name', type=str, default=None, help='which column to use, if not specified, use all encodable columns under the table')
parser.add_argument('--collection_name', type=str, default='text_bm25_en', help='which collection to use')
parser.add_argument('--limit', type=int, default=2, help='limit the number of returned results')
parser.add_argument('--agent_method', type=str, default='classic_rag', help='Agent method')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--max_turn', type=int, default=1, help='Maximum turns for the agent to interact with the environment')
parser.add_argument('--eval_llm', type=str, default='gpt-4o', help='Evaluation LLM model')
parser.add_argument('--eval_temperature', type=float, default=0.7, help='Evaluation temperature')
parser.add_argument('--eval_top_p', type=float, default=0.95, help='Evaluation top_p')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold for fuzzy matching during evaluation')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the interaction history')
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# the filename can be customized
filename = f'{args.dataset}_{args.agent_method}_{args.llm}-{start_time}'

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(os.path.join('logs', f'{filename}.log'), encoding='utf-8')
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
env: AgentEnv = ENVIRONMENTS['text2vec'](dataset=args.dataset, vectorstore=args.vectorstore)
agent: AgentBase = FRAMEWORKS['classic_rag'](llm, env, agent_method=args.agent_method)

test_data = []
if os.path.exists(args.test_data) and os.path.isfile(args.test_data):
    test_data_path = args.test_data
else: test_data_path = os.path.join('data', 'dataset', args.dataset, 'processed_data', args.test_data)
with open(test_data_path, 'r') as inf:
    for line in inf:
        test_data.append(json.loads(line))

preds = []
for data in test_data:
    logger.info(f"Processing question: {data['uuid']}")
    question, answer_format = formulate_input(args.vectorstore, data)
    result = agent.interact(
        question, answer_format,
        table_name=args.table_name, column_name=args.column_name,
        pdf_id=data['pdf_id'], page_number=data['page_number'],
        collection_name=args.collection_name, limit=args.limit,
        model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    preds.append({
        'uuid': data['uuid'],
        'question_type': data['question_type'],
        'answer': result
    })
logger.info(f"Total cost: {llm.get_cost()}")
agent.close()

output_path = os.path.join(args.result_dir, f'{filename}.jsonl')
with open(output_path, 'w') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")
result = evaluate(preds, test_data, args.dataset, model=args.eval_llm, temperature=args.eval_temperature, top_p=args.eval_top_p, threshold=args.threshold, verbose=False)
result_table = print_result(result)
logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

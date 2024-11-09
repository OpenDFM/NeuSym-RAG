#coding=utf8
import argparse, os, sys, json, logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.envs import ENVIRONMENTS
from agents.models import infer_model_class
from agents.frameworks import FRAMEWORKS
from agents.prompts import convert_database_schema_to_prompt, formulate_input
from utils.eval_utils import evaluate, print_result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pdfvqa', help='which dataset to use')
parser.add_argument('--database', type=str, default='biology_paper', help='which database to use')
parser.add_argument('--test_data', type=str, default='test_data_sample.jsonl', help='test data file')
parser.add_argument('--db_format', type=str, choices=['create_sql', 'detailed_json'], default='create_sql', help='Database schema serialization format')
parser.add_argument('--action_format', type=str, default='json', choices=['markdown', 'json', 'xml', 'yaml'], help='Action format for the environment')
parser.add_argument('--agent_method', type=str, default='react', help='Agent method')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--max_turn', type=int, default=10, help='Maximum turns for the agent to interact with the environment')
parser.add_argument('--window_size', type=int, default=3, help='History window size preserved in the prompt when calling LLMs')
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
filename = f'{args.dataset}_text2sql_{args.agent_method}_{args.action_format}_{args.llm}-{start_time}'

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


llm = infer_model_class(args.llm)()
env = ENVIRONMENTS['text2sql'](action_format=args.action_format, dataset=args.dataset, database=args.database)
agent = FRAMEWORKS['text2sql'](llm, env, agent_method=args.agent_method, max_turn=args.max_turn)

test_data = []
if os.path.exists(args.test_data) and os.path.isfile(args.test_data):
    test_data_path = args.test_data
else: test_data_path = os.path.join('data', 'dataset', args.dataset, 'processed_data', args.test_data)
with open(test_data_path, 'r') as inf:
    for line in inf:
        test_data.append(json.loads(line))

database_prompt = convert_database_schema_to_prompt(args.database, serialize_method=args.db_format)

start_time = datetime.now()
preds = []
for i, data in enumerate(test_data):
    logger.info(f"Processing question {i+1}: {data['uuid']}")
    question, answer_format = formulate_input(args.database, data)
    result = agent.interact(question, database_prompt, answer_format, window_size=args.window_size, model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    preds.append({
        'uuid': data['uuid'],
        'question_type': data['question_type'],
        'answer': result
    })
    current_cost = llm.get_cost()
    current_time = datetime.now()-start_time
    logger.info(f"[Statistics]: Current Cost: {current_cost} | Current Time: {current_time}")
    logger.info(f"[Estimate]: Total Cost: {current_cost / (i+1) * len(test_data)} | Total Time: {current_time / (i+1) * len(test_data)}")
logger.info(f"[Statistics]: Total Cost: {llm.get_cost()} | Total Time: {datetime.now()-start_time}")
agent.close()

output_path = os.path.join(args.result_dir, f'{filename}.jsonl')
with open(output_path, 'w') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")
result = evaluate(preds, test_data, args.dataset, model=args.eval_llm, temperature=args.eval_temperature, top_p=args.eval_top_p, threshold=args.threshold, verbose=False)
result_table = print_result(result)
logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")
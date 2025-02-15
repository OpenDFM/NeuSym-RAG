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
parser.add_argument('--dataset', type=str, required=True, help='which dataset to use')
parser.add_argument('--database', type=str, required=True, help='which database to use')
parser.add_argument('--database_path', type=str, help='Database path.')
parser.add_argument('--vectorstore', type=str, help='which vectorstore to use')
parser.add_argument('--vectorstore_path', type=str, help='Path to the vectorstore.')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
parser.add_argument('--db_format', type=str, choices=['create_sql', 'detailed_json'], default='create_sql', help='Database schema serialization format')
parser.add_argument('--image_limit', type=int, default=10, help='Maximum number of images to be shown in the agents response')
parser.add_argument('--agent_method', type=str, default='two_stage_text2sql', help='Agent method')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--max_turn', type=int, default=2, help='Maximum turns for the agent to interact with the environment')
parser.add_argument('--window_size', type=int, default=5, help='Window size for the agent to interact with the environment')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results')
args = parser.parse_args()

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'{args.dataset}_{args.agent_method}_{args.llm}-{start_time}'
if args.result_dir == "results":
    result_dir = os.path.join(args.result_dir, filename)
else:
    result_dir = args.result_dir
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


llm = infer_model_class(args.llm)()
env = ENVIRONMENTS['text2sql'](dataset=args.dataset, database=args.database, database_path=args.database_path)
agent = FRAMEWORKS['two_stage_text2sql'](llm, env, agent_method=args.agent_method)

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

database_prompt = convert_database_schema_to_prompt(args.database, serialize_method=args.db_format)

start_time = datetime.now()
preds = []

result_path = os.path.join(result_dir, 'result.jsonl')
if os.path.exists(result_path):
    with open(result_path, 'r', encoding='utf-8') as inf:
        for line in inf:
            preds.append(json.loads(line))
for data_idx, data in enumerate(test_data):
    for pred in preds:
        if pred['uuid'] == data['uuid']:
            with open(result_path, 'w', encoding='utf-8') as ouf:
                ouf.write(json.dumps(pred) + '\n')
            break
    else:
        logger.info(f"Processing question [{data_idx + 1}/{len(test_data)}]: {data['uuid']}")
        output_path = os.path.join(result_dir, f"{data['uuid']}.jsonl")
        try:
            result = agent.interact(args.dataset, data, database_prompt, model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, output_path=output_path, image_limit=args.image_limit)
        except Exception as e:
            logger.error(f"[❌Error❌]: ({data['uuid']}) {str(e)}")
            result = '[ERROR]: ' + str(e)
        with open(result_path, 'w', encoding='utf-8') as ouf:
            ouf.write(json.dumps({'uuid': data['uuid'], 'answer': result}) + '\n')
logger.info(f"[Statistics]: Total Cost: {llm.get_cost()} | Total Time: {datetime.now() - start_time} | Total Tokens: prompt {llm._prompt_tokens}, completion {llm._completion_tokens}")
agent.close()

if not args.no_eval:
    result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
    result_table = print_result(result)
    logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

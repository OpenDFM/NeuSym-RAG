#coding=utf8
import argparse, os, sys, json, logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.envs import ENVIRONMENTS, HybridEnv
from agents.models import infer_model_class
from agents.frameworks import FRAMEWORKS, HybridRAGAgent
from agents.prompts import convert_database_schema_to_prompt, convert_vectorstore_schema_to_prompt, formulate_input
from utils.eval_utils import evaluate, print_result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='airqa', help='which dataset to use')
parser.add_argument('--database', type=str, default='ai_research', help='which database to use')
parser.add_argument('--vectorstore', type=str, default='ai_research', help='which vectorstore to use')
parser.add_argument('--launch_method', type=str, default='standalone', choices=['standalone', 'docker'], help='launch method for vectorstore, chosen from ["docker", "standalone"]. Note that, for Windows OS, can only choose "docker".')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
parser.add_argument('--db_format', type=str, choices=['create_sql', 'detailed_json'], default='create_sql', help='Database schema serialization format')
parser.add_argument('--vs_format', type=str, choices=['detailed_json'], default='detailed_json', help='Vectorstore schema serialization format')
parser.add_argument('--action_format', type=str, default='markdown', choices=['markdown', 'json', 'xml', 'yaml'], help='Action format for the environment')
parser.add_argument('--output_format', type=str, default='json', choices=['markdown', 'string', 'html', 'json'], help='Output format for the environment execution results')
parser.add_argument('--agent_method', type=str, default='two_stage_hybrid', help='Agent method')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--max_turn', type=int, default=2, help='Maximum turns for the agent to interact with the environment')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
args = parser.parse_args()

if args.vectorstore is None: args.vectorstore = args.database
if args.database is None: args.database = args.vectorstore
assert args.database == args.vectorstore, f"Database and vectorstore must be the same, but got {args.database} and {args.vectorstore}, respectively."

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'{args.dataset}_hybrid_rag_{args.agent_method}_{args.action_format}_{args.output_format}_{args.llm}-{start_time}'
result_dir = os.path.join(args.result_dir, filename)
os.makedirs(result_dir, exist_ok=True)

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
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
env: HybridEnv = ENVIRONMENTS['hybrid'](action_format=args.action_format, agent_method=args.agent_method, dataset=args.dataset, database=args.database, vectorstore=args.vectorstore, launch_method=args.launch_method)
agent: HybridRAGAgent = FRAMEWORKS['two_stage_hybrid'](llm, env, agent_method=args.agent_method, max_turn=args.max_turn)

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
vectorstore_prompt = convert_vectorstore_schema_to_prompt(args.vectorstore, serialize_method=args.vs_format, add_description=False)

start_time = datetime.now()
preds = []
for data in test_data:
    logger.info(f"Processing question: {data['uuid']}")
    output_path = os.path.join(result_dir, f"{data['uuid']}.jsonl")
    result = agent.interact(args.dataset, data, database_prompt, vectorstore_prompt, model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, output_path=output_path, output_kwargs={'output_format': args.output_format}, with_vision=True)
    preds.append({'uuid': data['uuid'], 'answer': result})
logger.info(f"[Statistics]: Total Cost: {llm.get_cost()} | Total Time: {datetime.now() - start_time} | Total Tokens: prompt {llm._prompt_tokens}, completion {llm._completion_tokens}")
agent.close()

output_path = os.path.join(result_dir, 'result.jsonl')
with open(output_path, 'w', encoding='utf-8') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")
result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
result_table = print_result(result)
logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")
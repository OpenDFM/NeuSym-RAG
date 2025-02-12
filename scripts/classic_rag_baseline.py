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
parser.add_argument('--dataset', type=str, required=True, help='which dataset to use')
parser.add_argument('--vectorstore', type=str, required=True, help='which vectorstore to use')
parser.add_argument('--launch_method', type=str, default='standalone', choices=['standalone', 'docker'], help='launch method for vectorstore, chosen from ["docker", "standalone"].')
parser.add_argument('--database_path', type=str, help='Database path.')
parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='host + port for milvus started from docker')
parser.add_argument('--vectorstore_path', type=str, help='Path to the vectorstore.')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
parser.add_argument('--table_name', type=str, default='chunks', help='which table to use, if not specified, use all tables under the database')
parser.add_argument('--column_name', type=str, default='text_content', help='which column to use, if not specified, use all encodable columns under the table')
parser.add_argument('--collection_name', type=str, default='text_sentence_transformers_all_minilm_l6_v2', help='which collection to use')
parser.add_argument('--limit', type=int, default=4, help='limit the number of returned results')
parser.add_argument('--agent_method', type=str, default='classic_rag', help='Agent method')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--max_turn', type=int, default=1, help='Maximum turns for the agent to interact with the environment')
parser.add_argument('--image_limit', type=int, default=10, help='Maximum number of images to be shown in the agents response')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results')
args = parser.parse_args()

assert args.table_name is not None and args.column_name is not None, "Table name and column name must be specified."

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'{args.dataset}_{args.agent_method}_{args.llm}_{args.table_name}_{args.column_name}-{start_time}'
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
env: AgentEnv = ENVIRONMENTS['text2vec'](dataset=args.dataset, vectorstore=args.vectorstore, launch_method=args.launch_method, vectorstore_path=args.vectorstore_path, docker_uri=args.docker_uri, database_path=args.database_path)
agent: AgentBase = FRAMEWORKS['classic_rag'](llm, env, agent_method=args.agent_method)

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
for data in test_data:
    logger.info(f"Processing question: {data['uuid']}")
    output_path = os.path.join(result_dir, f"{data['uuid']}.jsonl")
    try:
        result = agent.interact(
            args.dataset, data,
            table_name=args.table_name, column_name=args.column_name, page_number=data.get('page_number', None),
            collection_name=args.collection_name, limit=args.limit,
            model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
            output_path=output_path, image_limit=args.image_limit
        )
    except Exception as e:
        result = '[ERROR]: ' + str(e)
    preds.append({'uuid': data['uuid'], 'answer': result})
logger.info(f"[Statistics]: Total Cost: {llm.get_cost()} | Total Time: {datetime.now() - start_time} | Total Tokens: prompt {llm._prompt_tokens}, completion {llm._completion_tokens}")
agent.close()

output_path = os.path.join(result_dir, 'result.jsonl')
with open(output_path, 'w', encoding='utf-8') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")

if not args.no_eval:
    result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
    result_table = print_result(result)
    logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

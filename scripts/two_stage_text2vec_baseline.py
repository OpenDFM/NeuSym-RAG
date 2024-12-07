#coding=utf8
import argparse, os, sys, json, logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.envs import ENVIRONMENTS
from agents.models import infer_model_class
from agents.frameworks import FRAMEWORKS, TwoStageText2VecRAGAgent
from agents.prompts import convert_vectorstore_schema_to_prompt, formulate_input
from utils.eval_utils import evaluate, print_result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='airqa', help='which dataset to use')
parser.add_argument('--vectorstore', type=str, default='ai_research', help='which vectorstore to use')
parser.add_argument('--launch_method', type=str, default='standalone', choices=['standalone', 'docker'], help='launch method for vectorstore, chosen from ["docker", "standalone"]. Note that, for Windows OS, can only choose "docker".')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
parser.add_argument('--vs_format', type=str, choices=['detailed_json'], default='detailed_json', help='Vectorstore schema serialization format')
parser.add_argument('--agent_method', type=str, default='two_stage_text2vec', help='Agent method')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--max_turn', type=int, default=2, help='Maximum turns for the agent to interact with the environment')
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
args = parser.parse_args()

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'{args.dataset}_{args.agent_method}_{args.llm}-{start_time}'
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


llm = infer_model_class(args.llm)()
env = ENVIRONMENTS['text2vec'](dataset=args.dataset, vectorstore=args.vectorstore, launch_method=args.launch_method)
agent: TwoStageText2VecRAGAgent = FRAMEWORKS['two_stage_text2vec'](llm, env, agent_method=args.agent_method)

test_data = []
if os.path.exists(args.test_data) and os.path.isfile(args.test_data):
    test_data_path = args.test_data
elif os.path.exists(os.path.join('data', 'dataset', args.dataset, args.test_data)):
    test_data_path = os.path.join('data', 'dataset', args.dataset, args.test_data)
else:
    test_data_path = os.path.join('data', 'dataset', args.dataset, 'processed_data', args.test_data)
with open(test_data_path, 'r') as inf:
    for line in inf:
        test_data.append(json.loads(line))

vectorstore_prompt = convert_vectorstore_schema_to_prompt(args.vectorstore, serialize_method=args.vs_format)

preds = []
for data in test_data:
    logger.info(f"Processing question: {data['uuid']}")
    question, answer_format = formulate_input(args.dataset, data)
    output_path = os.path.join(result_dir, f'{data["uuid"]}.jsonl')
    result = agent.interact(question, vectorstore_prompt, answer_format, model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, output_path=output_path)
    preds.append({'uuid': data['uuid'], 'answer': result})
logger.info(f"Total cost: {llm.get_cost()}")
agent.close()

output_path = os.path.join(result_dir, 'result.jsonl')
with open(output_path, 'w', encoding='UTF-8') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")
result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
result_table = print_result(result)
logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

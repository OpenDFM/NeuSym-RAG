#coding=utf8
import os, sys, json, yaml
from datetime import datetime
from logging import Logger
from argparse import Namespace
from typing import Dict, List, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.envs import infer_env_class, AgentEnv
from agents.models import infer_model_class, LLMClient
from agents.frameworks import infer_agent_class, AgentBase
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
from utils.hyperparam_utils import parse_args, get_result_folder, get_result_logger
from utils.graphrag_utils import config_graphrag_settings

args: Namespace = parse_args()
assert args.agent_method == 'iterative_graph_rag', "This script is only for Iterative Graph-RAG agent."
result_dir: str = get_result_folder(args)
logger: Logger = get_result_logger(result_dir)

# parser.add_argument('--api_key', type=str, default='OPENAI', choices=['OPENAI', 'EMPTY'], help='API key')
# parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='llm name')
# parser.add_argument('--llm_port', type=int, help='open-source llm port')
# parser.add_argument('--emb_model', type=str, default='bge-large-en-v1.5', help='open-source embedding model name')
# parser.add_argument('--emb_model_port', type=int, help='open-source embedding model port')

# for pdf_id in os.listdir(os.path.join(args.graphrag_root, 'graphs')):
#     with open(os.path.join(args.graphrag_root, 'graphs', pdf_id, '.env'), 'w', encoding='utf-8') as ouf:
#         ouf.write(f"GRAPHRAG_API_KEY={os.environ['OPENAI_API_KEY'] if args.api_key == 'OPENAI' else args.api_key}\n")
#     with open(os.path.join(args.graphrag_root, 'graphs', pdf_id, 'settings.yaml'), 'r', encoding='utf-8') as inf:
#         settings = yaml.safe_load(inf)
#     settings['llm']['model'] = args.llm
#     settings['llm']['api_base'] = os.environ['OPENAI_BASE_URL'].strip('/') if args.api_key == 'OPENAI' else f'http://localhost:{args.llm_port}/v1'
#     settings['embeddings']['llm']['model'] = 'text-embedding-3-small' if args.api_key == 'OPENAI' else args.emb_model
#     settings['embeddings']['llm']['api_base'] = os.environ['OPENAI_BASE_URL'].strip('/') if args.api_key == 'OPENAI' else f'http://localhost:{args.emb_model_port}/v1'
#     with open(os.path.join(args.graphrag_root, 'graphs', pdf_id, 'settings.yaml'), 'w', encoding='utf-8') as ouf:
#         yaml.safe_dump(settings, ouf)
# configure settings.yaml

config_graphrag_settings(args)

llm: LLMClient = infer_model_class(args.llm)(image_limit=args.image_limit, length_limit=args.length_limit)
env: AgentEnv = infer_env_class(args.agent_method)(dataset=args.dataset)
agent: AgentBase = infer_agent_class(args.agent_method)(llm, env, agent_method=args.agent_method, max_turn=args.max_turn)
test_data: List[Dict[str, Any]] = load_test_data(args.test_data, args.dataset)

start_time = datetime.now()
preds = []
for data_idx, data in enumerate(test_data):
    logger.info(f"Processing question [{data_idx + 1}/{len(test_data)}]: {data['uuid']}")
    output_path = os.path.join(result_dir, f"{data['uuid']}.jsonl")
    try:
        # question = data['question'].strip()
        # for turn in range(args.max_turn):
        #     command = [
        #         'graphrag', 'query',
        #         '--root', os.path.join(args.graphrag_root, 'graphs', data['anchor_pdf'][0]),
        #         '--method', 'local',
        #         '--query', f"[Question]: {question}\n[Answer Format]: {data['answer_format'].strip()}\nYou should wrap your answer in three backticks:\n```txt\nfinal answer\n```"
        #     ]
        #     process = subprocess.run(command, text=True, capture_output=True)
        #     print(process.stderr)
        #     result = process.stdout
        #     match = re.search(r"```txt(.*?)```", result.strip())
        #     if match:
        #         result = match.group(1).strip()
        #     else:
        #         result = ""
        #     if turn == args.max_turn - 1:
        #         break
        #     response = llm.get_response(
        #         messages=[
        #             {
        #                 'role': 'system',
        #                 'content': 'You are a helpful assistant. Now you need to determine whether the answer is valid for the question. If valid, please output only 1 word "valid". If not valid, please refine the question and output it in three backticks:\n```txt\nrefined question\n```'
        #             },
        #             {
        #                 'role': 'user',
        #                 'content': f"[Question]: {question}\n[Answer]: {result}"
        #             }
        #         ],
        #         model=args.llm
        #     )
        #     if response.strip().lower() == 'valid':
        #         break
        #     match = re.search(r"```txt(.*?)```", response.strip())
        #     if match:
        #         question = match.group(1).strip()
        result = agent.interact(
            args.dataset, data,
            graphrag_root=args.graphrag_root, graphrag_method=args.graphrag_method,
            output_path=output_path
        )
        logger.info(f"✅✅✅✅✅ -> [{data['uuid']}]: {result}")
    except Exception as e:
        result = '[ERROR]: ' + str(e)
        logger.error(f"❌❌❌❌❌ -> [{data['uuid']}]: {str(e)}")
    preds.append({'uuid': data['uuid'], 'answer': result})

output_path = os.path.join(result_dir, 'result.jsonl')
write_jsonl(preds, output_path)
logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")

logger.info(f"[Statistics]: Total Cost: {llm.get_cost()} | Total Time: {datetime.now() - start_time} | Total Tokens: prompt {llm._prompt_tokens}, completion {llm._completion_tokens}")

if not args.no_eval:
    result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
    result_table = print_result(result)
    logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

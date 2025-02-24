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
from utils.graphrag_utils import load_yaml, write_yaml

args: Namespace = parse_args()
assert args.agent_method == 'two_stage_graph_rag', "This script is only for Two-stage Graph-RAG agent."
result_dir: str = get_result_folder(args)
logger: Logger = get_result_logger(result_dir)

# configure settings.yaml
settings_path = os.path.join(args.graphrag_root, 'settings.yaml')
settings = load_yaml(settings_path)
settings['llm']['model'] = args.llm
write_yaml(settings, settings_path)

llm: LLMClient = infer_model_class(args.llm)(image_limit=args.image_limit, length_limit=args.length_limit)
env: AgentEnv = infer_env_class(args.agent_method)(
    dataset=args.dataset,
    action_format=args.action_format,
    interact_protocol=args.interact_protocol
)
agent: AgentBase = infer_agent_class(args.agent_method)(llm, env, agent_method=args.agent_method)
test_data: List[Dict[str, Any]] = load_test_data(args.test_data, args.dataset)

start_time = datetime.now()
preds = []
for data_idx, data in enumerate(test_data):
    logger.info(f"Processing question [{data_idx + 1}/{len(test_data)}]: {data['uuid']}")
    output_path = os.path.join(result_dir, f"{data['uuid']}.jsonl")
    try:
        # graphrag_query = f"[Question]: {data['question'].strip()}\n[Answer Format]: {data['answer_format'].strip()}\nYou should wrap your answer in three backticks:\n```txt\nfinal answer\n```"
        # if len(data['anchor_pdf']) > 0:
        #     anchor_pdf_titles = []
        #     for anchor_pdf_id in data['anchor_pdf']:
        #         with open(os.path.join('data', 'dataset', args.dataset, 'metadata', anchor_pdf_id + '.json'), 'r', encoding='utf-8') as inf:
        #             anchor_pdf_titles.append(json.load(inf)['title'].replace('\n', ' ').strip())
        #     anchor_pdf_titles_str = '\n'.join(anchor_pdf_titles)
        #     graphrag_query += f"\n[Related Paper Titles]:\n{anchor_pdf_titles_str}"
        # command = [
        #     'graphrag', 'query',
        #     '--root', args.graphrag_root,
        #     '--method', 'global' if 'retrieval' in data['tags'] else 'local',
        #     '--query', graphrag_query
        # ]
        # process = subprocess.run(command, text=True, capture_output=True)
        # result = process.stdout
        # match = re.search(r"```txt(.*?)```", result.strip())
        # if match:
        #     result = match.group(1).strip()
        # else:
        #     result = ""
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

agent.close()
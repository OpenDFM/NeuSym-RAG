#coding=utf8
import os, sys
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
assert args.agent_method == 'two_stage_graph_rag', "This script is only for Two-stage Graph-RAG agent."
result_dir: str = get_result_folder(args)
logger: Logger = get_result_logger(result_dir)

llm: LLMClient = infer_model_class(args.llm)(image_limit=args.image_limit, length_limit=args.length_limit)
env: AgentEnv = infer_env_class(args.agent_method)(
    dataset=args.dataset,
    action_format=args.action_format,
    interact_protocol=args.interact_protocol
)
agent: AgentBase = infer_agent_class(args.agent_method)(llm, env, agent_method=args.agent_method)
test_data: List[Dict[str, Any]] = load_test_data(args.test_data, args.dataset)

# configure settings.yaml, overwrite the llm and embedding model
graphrag_root = config_graphrag_settings(args.dataset, args.llm, args.graphrag_embed)

start_time = datetime.now()
preds = []
for data_idx, data in enumerate(test_data):
    logger.info(f"Processing question [{data_idx + 1}/{len(test_data)}]: {data['uuid']}")
    output_path = os.path.join(result_dir, f"{data['uuid']}.jsonl")
    try:
        result = agent.interact(
            args.dataset, data,
            graphrag_root=graphrag_root, graphrag_method=args.graphrag_method,
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

logger.info(f"[Statistics]: Total Time: {datetime.now() - start_time}")

if not args.no_eval:
    result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
    result_table = print_result(result)
    logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

agent.close()
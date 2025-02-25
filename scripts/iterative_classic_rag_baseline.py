#coding=utf8
import os, sys, json
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

args: Namespace = parse_args()
assert args.agent_method == 'iterative_classic_rag', "This script is only for Iterative Classic-RAG agent."
result_dir: str = get_result_folder(args)
logger: Logger = get_result_logger(result_dir)

llm: LLMClient = infer_model_class(args.llm)(image_limit=args.image_limit, length_limit=args.length_limit)
env: AgentEnv = infer_env_class(args.agent_method)(
    dataset=args.dataset,
    action_format=args.action_format,
    interact_protocol=args.interact_protocol,
    vectorstore=args.vectorstore,
    database_path=args.database_path,
    launch_method=args.launch_method,
    vectorstore_path=args.vectorstore_path,
    docker_uri=args.docker_uri
)
agent: AgentBase = infer_agent_class(args.agent_method)(llm, env, agent_method=args.agent_method, max_turn=args.max_turn)
test_data: List[Dict[str, Any]] = load_test_data(args.test_data, args.dataset)

start_time = datetime.now()
preds = []
for data_idx, data in enumerate(test_data):
    logger.info(f"Processing question [{data_idx + 1}/{len(test_data)}]: {data['uuid']}")
    output_path = os.path.join(result_dir, f'{data["uuid"]}.jsonl')
    try:
        result = agent.interact(
            args.dataset, data,
            collection_name=args.collection_name, table_name=args.table_name, column_name=args.column_name,
            window_size=args.window_size,
            model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, 
            output_kwargs={'output_format': args.output_format}, output_path=output_path
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
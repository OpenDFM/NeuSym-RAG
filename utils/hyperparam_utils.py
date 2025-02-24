#coding=utf8
from typing import List, Dict, Any, Optional
import os, json, sys, datetime
import argparse, logging


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset, database, vectorstore utils
    parser.add_argument('--dataset', type=str, required=True, help='Which dataset to use.')
    parser.add_argument('--database', type=str, help='Which database to use, i.e., the name of the DB.')
    parser.add_argument('--database_path', type=str, help='Database path.')
    parser.add_argument('--database_type', type=str, default='duckdb', help='Which database type to use. We only support DuckDB currently.')
    parser.add_argument('--vectorstore', type=str, help='Which vectorstore to use, usually the same name with the database.')
    parser.add_argument('--launch_method', type=str, default='standalone', choices=['standalone', 'docker'], help='Launch method for vectorstore, chosen from ["docker", "standalone"]. `standalone` -> from `.db` file; `docker` -> from docker containers.')
    parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='The host:port for vectorstore started from docker.')
    parser.add_argument('--vectorstore_path', type=str, help='Path to the vectorstore if launched from method `standalone`.')
    parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='Test data file or path. If file name, search the default filepath `data/dataset/${dataset}/${test_data}`.')

    # agent, llm, env utils
    parser.add_argument('--db_format', type=str, choices=['create_sql', 'detailed_json'], default='create_sql', help='Database schema serialization format. See agents/prompts/schema_prompt.py for details.')
    parser.add_argument('--vs_format', type=str, choices=['detailed_json'], default='detailed_json', help='Vectorstore schema serialization format. See agents/prompts/schema_prompt.py for details.')
    parser.add_argument('--action_format', type=str, default='markdown', choices=['markdown', 'json', 'xml', 'yaml'], help='Action format for the environment acceptable inputs.')
    parser.add_argument('--output_format', type=str, default='json', choices=['markdown', 'json', 'html', 'string'], help='Output/Observation format of tables for the environment execution results.')
    parser.add_argument('--agent_method', type=str, default='react', choices=[
        'trivial_question_only', 'trivial_title_with_abstract', 'trivial_full_text_with_cutoff', 'classic_rag', 'react'
    ], help='Various agent / baseline method.')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='LLM name to use. See agents/models for all supported LLMs.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling from the LLM.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p for sampling from the LLM.')
    parser.add_argument('--max_tokens', type=int, default=1500, help='Maximum number of tokens to generate, a.k.a., the maximum completion tokens.')
    parser.add_argument('--max_turn', type=int, default=20, help='Maximum turns for the agent to interact with the environment.')
    parser.add_argument('--window_size', type=int, default=5, help='History window size, or the number of previous (action, observation) pairs preserved in the prompt when calling LLMs.')
    parser.add_argument('--image_limit', type=int, default=10, help='Maximum number of images to be shown in the agent prompt. Also restricted by the LLMs/VLMs, e.g., --limit_mm_per_prompt.')
    parser.add_argument('--length_limit', type=int, default=32, help='The total length limit of the prompt (multiplied by 1000). By default, 32k.')

    # method specific hyperparams
    parser.add_argument('--table_name', type=str, default='chunks', help='For Classic-RAG, the table name to retrieve context.')
    parser.add_argument('--column_name', type=str, default='text_content', help='For Classic-RAG, the column name to retrieve context.')
    parser.add_argument('--collection_name', type=str, default='text_bm25_en', help='For Classic-RAG, the collection name to retrieve context.')
    parser.add_argument('--limit', type=int, default=4, help='For Classic-RAG, the limit or top K of the retrieved chunks.')
    parser.add_argument('--cutoff', type=int, default=5, help='For full-text with cutoff baseline, restrict the length of tokens (multiply 1000) for the full-text.')
    # output, result utils
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
    parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results, because subjective evaluation usually involves LLM-based judgement.')
    args = parser.parse_args()

    # validate consistency for hyperparams with different methods
    validate_args(args)
    return args


def validate_args(args):
    """ Validate the argument consistency for different agent methods to ensure consistency.
    """
    if args.agent_method in ['trivial_title_with_abstract', 'trivial_full_text_with_cutoff']:
        assert args.cutoff > 0, "Cutoff must be greater than 0 for trivial title with abstract and full-text with cutoff agent."
    elif args.agent_method == 'classic_rag':
        assert args.table_name is not None and args.column_name is not None, "Table name and column name must be specified for Classic-RAG agent."
        assert args.collection_name is not None, "Collection name must be specified for Classic-RAG agent."
        assert args.limit > 0, "Limit must be greater than 0 for Classic-RAG agent."
    return args


def get_result_folder(args) -> str:
    """ Get the complete path to the result folder, auto-constructed by the arguments.
    """
    root_result_dir = args.result_dir
    split_index = result_dir = ''
    # parallel run multiple test data, see utils/dataset_utils.py
    if 'split_' in args.test_data:
        split_index = '_split' + args.test_data.split('.')[0].split('_')[-1]

    # customize the result folder name
    result_dir = f'{args.dataset}{split_index}_{args.agent_method}_{args.llm}'
    if args.agent_method in ['trivial_title_with_abstract', 'trivial_full_text_with_cutoff']:
        result_dir += f"_cutoff-${args.cutoff}"
    elif args.agent_method == 'classic_rag':
        result_dir += f"_{args.table_name}_{args.column_name}_{args.collection_name}_{args.limit}"

    # add timestamp suffix to the result folder
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir += f'-{start_time}'
    result_dir = os.path.join(root_result_dir, result_dir)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def get_result_logger(result_dir: str, name: str = 'experiments') -> str:
    """ Get the logger for experiments. Write to log file and stdout at the same time.
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(result_dir, 'log.txt'), encoding='utf-8')
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)s|%(lineno)d][%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger
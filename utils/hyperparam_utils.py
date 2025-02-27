#coding=utf8
from typing import List, Dict, Any, Optional
import os, json, sys
from datetime import datetime
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
    parser.add_argument('--agent_method', type=str, default='neusym_rag', choices=[
        'trivial_question_only', 'trivial_title_with_abstract', 'trivial_full_text_with_cutoff', 'classic_rag', 'two_stage_neu_rag', 'two_stage_sym_rag', 'two_stage_graph_rag', 'two_stage_hybrid_rag', 'iterative_classic_rag', 'iterative_neu_rag', 'iterative_sym_rag', 'iterative_graph_rag', 'neusym_rag'
    ], help='Various agent / baseline method.')
    parser.add_argument('--interact_protocol', type=str, default='react', choices=['react', 'code_block'], help='Interaction protocol for the agent method which is used to extract the parsable action text from LLM response, chosen from ["react", "code_block"].')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='LLM name to use. See agents/models for all supported LLMs.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling from the LLM.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p for sampling from the LLM.')
    parser.add_argument('--max_tokens', type=int, default=1500, help='Maximum number of tokens to generate, a.k.a., the maximum completion tokens.')
    parser.add_argument('--max_turn', type=int, default=20, help='Maximum turns for the agent to interact with the environment.')
    parser.add_argument('--window_size', type=int, default=5, help='History window size, or the number of previous (action, observation) pairs preserved in the prompt when calling LLMs.')
    parser.add_argument('--image_limit', type=int, default=10, help='Maximum number of images to be shown in the agent prompt. Also restricted by the LLMs/VLMs, e.g., --limit_mm_per_prompt.')
    parser.add_argument('--length_limit', type=int, default=32, help='The total length limit of the prompt (multiplied by 1000). By default, 32k.')

    # method specific hyperparams
    parser.add_argument('--collection_name', type=str, default='text_bm25_en', help='For Classic-RAG and Iterative Classic-RAG methods, the collection name to retrieve context.')
    parser.add_argument('--table_name', type=str, default='chunks', help='For Classic-RAG and Iterative Classic-RAG methods, the table name to retrieve context.')
    parser.add_argument('--column_name', type=str, default='text_content', help='For Classic-RAG and Iterative Classic-RAG methods, the column name to retrieve context.')
    parser.add_argument('--limit', type=int, default=4, help='For Classic-RAG, the limit or top K of the retrieved chunks.')
    parser.add_argument('--cutoff', type=int, default=5, help='For title with abstract and full-text with cutoff baseline, restrict the length of tokens (multiply 1000) for the full-text.')
    parser.add_argument('--graphrag_root', type=str, default='', help='For Graph-RAG and Iterative Graph-RAG, the root folder, which should contains settings.yaml.')
    parser.add_argument('--graphrag_method', type=str, default='local', choices=['local', 'global'], help='For Graph-RAG and Iterative Graph-RAG, the method to use, chosen from ["local", "global"].')

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
        assert args.cutoff > 0, "Cutoff must be greater than 0 for trivial input of title with abstract or full-text with cutoff agent."
    elif args.agent_method in ['classic_rag', 'iterative_classic_rag']:
        assert args.vectorstore is not None, "Vectorstore must be specified for Classic-RAG or Iterative Classic-RAG agent."
        assert args.table_name is not None and args.column_name is not None, "Table name and column name must be specified for Classic-RAG or Iterative Classic-RAG agent."
        assert args.collection_name is not None, "Collection name must be specified for Classic-RAG or Iterative Classic-RAG agent."
    elif args.agent_method in ['two_stage_hybrid_rag', 'neusym_rag']:
        assert args.database or args.vectorstore, "At least database or vectorstore must be specified for Two-stage Hybrid-RAG or NeuSym-RAG agent."
        if args.vectorstore is None: args.vectorstore = args.database
        if args.database is None: args.database = args.vectorstore
        assert args.database == args.vectorstore, f"Database and vectorstore must be the same, but got {args.database} and {args.vectorstore}, respectively."
    elif args.agent_method in ['two_stage_neu_rag', 'iterative_neu_rag']:
        assert args.vectorstore is not None, "Vectorstore must be specified for Two-stage Neu-RAG or Iterative Neu-RAG agent."
    elif args.agent_method in ['two_stage_sym_rag', 'iterative_sym_rag']:
        assert args.database is not None, "Database must be specified for Two-stage Sym-RAG or Iterative Sym-RAG agent."
    elif args.agent_method in ['two_stage_graph_rag', 'iterative_graph_rag']:
        assert args.graphrag_root and os.path.exists(args.graphrag_root) and os.path.isdir(args.graphrag_root), "Graph-RAG root folder must be specified and exist for Two-stage Graph-RAG or Iterative Graph-RAG agent."
    
    if args.agent_method.startswith('trivial') or args.agent_method.startswith('two_stage') or args.agent_method in ['classic_rag', 'iterative_graph_rag']:
        # assert args.interact_protocol == 'code_block', "`code_block` interact protocol is required for Trivial Baselines, Two-stage Hybrid-RAG, Two-stage Neu-RAG, Two-stage Sym-RAG, Two-stage Graph-RAG, Classic-RAG and Iterative Graph-RAG agents."
        args.interact_protocol = 'code_block'
    if args.agent_method in ['iterative_classic_rag', 'iterative_neu_rag', 'iterative_sym_rag', 'neusym_rag']:
        args.interact_protocol = 'react'
    if args.agent_method.startswith('two_stage'):
        args.action_format = 'json' # for easier parsing
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
    elif args.agent_method in ['classic_rag', 'iterative_classic_rag']:
        result_dir += f"_{args.collection_name}_{args.table_name}_{args.column_name}"
        if args.agent_method == 'classic_rag':
            result_dir += f"_limit-{args.limit}"
    elif 'graphrag' in args.agent_method:
        result_dir += f"_{args.graphrag_method}"
    if args.agent_method.startswith('iterative') or args.agent_method == 'neusym_rag':
        result_dir += f"_action_{args.action_format}_output_{args.output_format}"
        result_dir += f"_turn_{args.max_turn}_window_{args.window_size}"

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
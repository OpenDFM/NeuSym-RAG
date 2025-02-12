#coding=utf8
import argparse, os, sys, json, logging, re
from datetime import datetime
from typing import Dict, List, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.models import infer_model_class, LLMClient
from utils.eval_utils import evaluate, print_result
from utils.functions.ai_research_metadata import get_airqa_paper_metadata

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='which dataset to use')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='test data file')
parser.add_argument('--llm', type=str, default='gpt-4o-mini')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--max_tokens', type=int, default=1500)
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results')
args = parser.parse_args()

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'{args.dataset}_question_only_{args.llm}-{start_time}'
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
airqa_metadata = get_airqa_paper_metadata()

SYSTEM_PROMPT = "You are intelligent agent who is expert in answering user questions based on the retrieved context. You will be given a natural language question concerning several PDF files, and your task is to answer the input question with predefined output format using the relevant information."

CONTEXT_PROMPT = """[Title and Abstract of {index}]:
title: ```{title}```
abstract: ```{abstract}```"""

AGENT_PROMPT = """Here is the task input:

[Question]: {question}
[Answer Format]: {answer_format}
{context}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER WITH REQUIRED FORMAT IN THREE BACKTICKS):

```txt\nfinal answer\n```
"""

for data in test_data:
    prev_cost = llm.get_cost()
    logger.info(f"Processing question: {data['uuid']}")
    question, answer_format = data["question"], data["answer_format"]
    logger.info(f'[Question]: {question}')
    logger.info(f'[Answer Format]: {answer_format}')
    anchor_pdf, reference_pdf = data["anchor_pdf"], data["reference_pdf"]
    context = ""
    for idx, uuid in enumerate(anchor_pdf, start=1):
        context += CONTEXT_PROMPT.format(index=f"Anchor PDF {idx}", title=airqa_metadata[uuid]['title'], abstract=airqa_metadata[uuid]['abstract']) + "\n"
    for idx, uuid in enumerate(reference_pdf, start=1):
        context += CONTEXT_PROMPT.format(index=f"Reference PDF {idx}", title=airqa_metadata[uuid]['title'], abstract=airqa_metadata[uuid]['abstract']) + "\n"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": AGENT_PROMPT.format(question=question, answer_format=answer_format, context=context)
        }
    ]
    logger.info('Generate Answer ...')
    response = llm.get_response(messages, model=args.llm, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    logger.info(f'[Response]: {response}')
    matched_list = re.findall(r"```(txt)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL)
    if not matched_list:
        result = response.strip()
    else:
        result = matched_list[-1][1].strip()
    logger.info(f'[Answer]: {result}')

    cost = llm.get_cost() - prev_cost
    logger.info(f'[Info]: LLM API call costs ${cost:.6f}.')
    preds.append({'uuid': data['uuid'], 'answer': result})
logger.info(f"[Statistics]: Total Cost: {llm.get_cost()} | Total Time: {datetime.now() - start_time} | Total Tokens: prompt {llm._prompt_tokens}, completion {llm._completion_tokens}")

output_path = os.path.join(result_dir, 'result.jsonl')
with open(output_path, 'w', encoding='utf-8') as ouf:
    for pred in preds:
        ouf.write(json.dumps(pred) + '\n')
    logger.info(f"{len(preds)} predictions on {args.dataset} saved to {output_path}")

if not args.no_eval:
    result = evaluate(preds, test_data, args.dataset, output_path=os.path.join(result_dir, 'evaluation.txt'))
    result_table = print_result(result)
    logger.info(f"Final evaluation result on {args.dataset}:\n{result_table}")

#coding=utf8
import sys, os, json
import tiktoken
from tiktoken import Encoding
from typing import List, Dict, Any, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents.envs.actions import GenerateAnswer, Action


encoding_models = dict()

def load_jsonl(fp: str) -> list:
    with open(fp, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def truncate_tokens(text: str, max_tokens: int = 30, encoding_model: str = 'cl100k_base') -> str:
    """ Given a text string, truncate it to max_tokens using encoding_model tokenizer
    """
    global encoding_models
    if encoding_model not in encoding_models:
        encoding: Encoding = tiktoken.get_encoding(encoding_model)
        encoding_models[encoding_model] = encoding
    encoding: Encoding = encoding_models[encoding_model]
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens * 1000:
        tokens = tokens[:max_tokens * 1000]
        text = encoding.decode(tokens)
    return text


def extract_answer_from_trajectory(data: List[Dict[str, Any]], action_format: str = 'markdown', agent_method: str = 'react') -> Any:
    trailing_count = 2 # only check the last two messages, o.w., no answer
    for msg in data[len(data) - 1:len(data) - 1 - trailing_count:-1]:
        if '[Action]: GenerateAnswer(' in msg['content']:
            flag, action = Action.parse_action(
                msg['content'],
                action_types=[GenerateAnswer], # only one action type
                action_format=action_format,
                agent_method=agent_method
            )
            if flag:
                result = action.execute(None).obs_content
                result = truncate_tokens(str(result))
                return result
    result = '[ERROR]: No answer found.'
    return result
    

def generate_result_from_trajectory(
        pred_folder: str,
        gold: str,
        action_format: str = 'markdown',
        agent_method: str = 'react',
        force: bool = False,
        **kwargs
    ) -> dict:
    output_path = os.path.join(pred_folder, 'result.jsonl')
    if not force and os.path.exists(output_path):
        print('[WARNING]: The result.jsonl already exists, please ensure that you want to extract the result again.')
        return load_jsonl(output_path)

    gold_data = load_jsonl(gold)
    gold_uuids = [data['uuid'] for data in gold_data]

    results, failed_count, failed_uuids = {}, 0, []
    for file in os.listdir(pred_folder):
        if file.endswith('.jsonl') and file != 'result.jsonl':
            pred_path = os.path.join(pred_folder, file)
            data = load_jsonl(pred_path)
            try:
                answer = extract_answer_from_trajectory(data, action_format=action_format, agent_method=agent_method)
            except Exception as e:
                answer = f'[ERROR]: {str(e)}'
            qid = file.split('.')[0]
            if isinstance(answer, str) and answer == '[ERROR]: No answer found.':
                failed_count += 1
                failed_uuids.append(qid)
            results[qid] = {
                'uuid': qid,
                'answer': answer.encode('utf-8', 'ignore').decode('utf-8')
            }
    print(f"[INFO]: Failed UUIDs: {failed_uuids}")
    print(f"[INFO]: Failed to extract answer for {failed_count}/{len(results)} questions.")
    with open(output_path, 'w', encoding='utf8') as f:
        for uid in gold_uuids:
            f.write(json.dumps(results[uid], ensure_ascii=False) + '\n')
    print(f"{len(results)} predictions saved to {output_path}")
    return results


def calculate_failed_ratio(result_folder: str):
    results = load_jsonl(os.path.join(result_folder, 'result.jsonl'))
    failed_count = 0
    for res in results:
        if res['answer'].startswith('[ERROR]:'):
            failed_count += 1
    print(f"Failed to extract answer for {failed_count}/{len(results)} questions.")
    return failed_count / len(results)



if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=str, required=True, help='The folder containing the prediction files.')
    parser.add_argument('--gold', type=str, help='The gold file path for sort.')
    parser.add_argument('--action_format', type=str, default='markdown', help='The format of the action.')
    parser.add_argument('--agent_method', type=str, default='react', help='The method of the agent.')
    parser.add_argument('--failed_ratio', action='store_true', help='Calculate the failed ratio.')
    parser.add_argument('--force', action='store_true', help='Force to extract the result again.')
    args = parser.parse_args()

    if args.failed_ratio:
        calculate_failed_ratio(args.pred_folder)
        sys.exit(0)

    generate_result_from_trajectory(
        args.pred_folder,
        args.gold,
        action_format=args.action_format,
        agent_method=args.agent_method,
        force=args.force
    )
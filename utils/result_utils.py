#coding=utf8
import sys, os, re, json
import tiktoken, re, shutil
from tiktoken import Encoding
from typing import List, Dict, Any, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents.envs.actions import GenerateAnswer, Action
from agents.frameworks import truncate_tokens


def load_jsonl(fp: str) -> list:
    with open(fp, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    return


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


def calculate_interaction_turn(result_folder: str):
    turns = []
    for fp in os.listdir(result_folder):
        if fp.endswith('.jsonl') and fp != 'result.jsonl':
            data = load_jsonl(os.path.join(result_folder, fp))
            turns.append(len(data) // 2 - 1)
    print(f'Average interaction turn: {sum(turns) / len(turns)}')
    return turns


def extract_failed_uuids(result_folder: str) -> List[str]:
    failed_uuids = []
    with open(os.path.join(result_folder, 'evaluation.txt'), 'r', encoding='utf8') as f:
        for line in f:
            if '[ERROR]: data' in line:
                # with id 00608f20-e3f5-5fdc-8979-4efeb0756d8e
                uid = re.search(r'with id ([0-9a-z\-]+)\s*$')
                if uid:
                    failed_uuids.append(uid.group(1))
                else:
                    print(f'[ERROR]: Failed to extract uuid from {line}')
    with open(result_folder, 'failed_uuids.json', 'w', encoding='utf8') as f:
        json.dump(failed_uuids, f, indent=4)
    return failed_uuids


def extract_gold_result(result_folder: str, gold: str):
    target_data = []
    gold_data = load_jsonl(gold)
    gold_uuids = [data['uuid'] for data in gold_data]
    pred_data = load_jsonl(os.path.join(result_folder, 'result.jsonl'))
    pred_data = {data['uuid']: data for data in pred_data}
    for uid in gold_uuids:
        assert uid in pred_data, f'[ERROR]: {uid} not found in {args.pred_folder}'
        target_data.append(pred_data[uid])
    write_jsonl(target_data, os.path.join(result_folder, 'new_result.jsonl'))
    return


def merge_result(gold: str, keywords: List[str] = [], result_folder: str = 'results/'):
    if not result_folder: result_folder = 'results/'
    filtered_folders = []
    keywords = ['_split'] + [k for k in keywords if k not in ['_split']]
    gold_data = load_jsonl(gold)
    for subfolder in os.listdir(result_folder):
        if any(k not in subfolder for k in keywords): continue
        filtered_folders.append(subfolder)
    try:
        indexes = []
        for folder in filtered_folders:
            # print('Extract index for', folder)
            sid = int(re.search(r'_split(\d+)_', folder).group(1))
            indexes.append(sid)
    except Exception as e:
        print(f'[ERROR]: Failed to extract index from subfolder: {e}')
        return

    sorted_indexes = sorted(indexes)
    if list(range(len(indexes))) != sorted_indexes:
        print(f'[ERROR]: The indexes are not continuous (0,1,2,...) or duplicate indexes occurred. Get {sorted_indexes}\nPlease delete redundant folders or add morekeywords to filter subfolders.')
        return
    sorted_folders = sorted(zip(indexes, filtered_folders), key=lambda x: x[0])
    merged_data = []
    for _, subfolder in sorted_folders:
        data = load_jsonl(os.path.join(result_folder, subfolder, 'result.jsonl'))
        merged_data.extend(data)

    # final check: uuid sorted
    merged_uuids = [data['uuid'] for data in merged_data]
    gold_uuids = [data['uuid'] for data in gold_data]
    if len(merged_uuids) != len(gold_uuids):
        print(f'[ERROR]: Total number of predicted examples are not correct! Get {len(merged_uuids)} pred v.s. {len(gold_uuids)} gold.')
        return
    if merged_uuids != gold_uuids:
        print(f'[ERROR]: The uuids are not sorted according to {gold}. Please check the results.')
        return

    # safely move files
    target_folder = os.path.join(result_folder, f'merged{"_".join(keywords)}')
    os.makedirs(target_folder, exist_ok=True)
    for _, subfolder in sorted_folders:
        folder = os.path.join(result_folder, subfolder)
        for fp in os.listdir(folder):
            if fp.endswith('.jsonl') and fp.split('.')[0] in gold_uuids:
                shutil.copy(os.path.join(folder, fp), os.path.join(target_folder, fp))

    write_jsonl(merged_data, os.path.join(target_folder, 'result.jsonl'))
    print(f'Successfully merged {len(merged_data)} data in {len(sorted_folders)} sub-folders to {target_folder} !')
    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='The folder containing the prediction files.')
    parser.add_argument('--gold', type=str, help='The gold file path for sort.')
    parser.add_argument('--action_format', type=str, default='markdown', help='The format of the action.')
    parser.add_argument('--agent_method', type=str, default='react', help='The method of the agent.')
    parser.add_argument('--keywords', type=str, nargs='*', help='The keywords to filter subfolders.')
    parser.add_argument('--function', type=str, default='extract_answer', choices=['calc_error_ratio', 'calc_num_turns', 'extract_failed_uuids', 'extract_answer', 'extract_gold_result', 'merge_result'], help='Calculate the failed ratio.')
    parser.add_argument('--force', action='store_true', help='Force to extract the result again.')
    args = parser.parse_args()

    if args.function == 'calc_error_ratio':
        calculate_failed_ratio(args.folder)
    elif args.function == 'calc_num_turns':
        calculate_interaction_turn(args.folder)
    elif args.function == 'extract_failed_uuids':
        extract_failed_uuids(args.folder)
    elif args.function == 'extract_gold_result':
        extract_gold_result(args.folder, args.gold)
    elif args.function == 'merge_result':
        merge_result(args.gold, args.keywords, args.folder)
    else:
        generate_result_from_trajectory(
            args.folder,
            args.gold,
            action_format=args.action_format,
            agent_method=args.agent_method,
            force=args.force
        )

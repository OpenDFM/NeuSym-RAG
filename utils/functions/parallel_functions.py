#coding=utf8
import json, hashlib, sys, os, tiktoken
from tqdm import tqdm
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.functions.common_functions import convert_to_message, call_llm_with_message

def hashed(stringified_message: str) -> str:
    return hashlib.md5(stringified_message.encode("utf-8")).hexdigest()

PARALLEL_DICT = {}

def truncate_tokens(text: str, max_tokens: int = 30, encoding_model: str = 'cl100k_base') -> str:
    """ Given a text string, truncate it to max_tokens using encoding_model tokenizer
    """
    encoding = tiktoken.get_encoding(encoding_model)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) > max_tokens * 1000:
        tokens = tokens[:max_tokens * 1000]
        text = encoding.decode(tokens)
    return text

def parallel_write_or_read(
        template: str,
        **kwargs
    ) -> str:
    stringified_message = json.dumps(convert_to_message(truncate_tokens(template), **kwargs), separators=(",", ":"))
    hashed_message = hashed(stringified_message.strip())
    parallel = kwargs.get("parallel")
    parallel_dict = {}
    if parallel.get("read"):
        if PARALLEL_DICT.get(parallel["read"], {}) == {}:
            PARALLEL_DICT[parallel["read"]] = json.load(open(parallel["read"], "r", encoding='utf-8'))
        parallel_dict = PARALLEL_DICT[parallel["read"]]
        if parallel_dict.get(hashed_message):
            return parallel_dict[hashed_message]
        print(f"Message {hashed_message} not found in the parallel file.")
    if parallel.get("write"):
        with open(parallel["write"], "a", encoding='utf-8') as f:
            f.write(stringified_message + "\n")
        return ""
    return "NO SUMMARY"

def parallel_message_to_batch(
        message_group: List[List[Dict[str, str]]], 
        hashed_group: List[str],
        model: str = 'qwen2-vl-72b-instruct',
        max_tokens: int = 1500
    ):
    batch_group = []
    for message, hashed_message in zip(message_group, hashed_group):
        batch_group.append(
            {
                "custom_id": f"{hashed_message}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": message,
                    "max_tokens": max_tokens
                }
            }
        )
    return batch_group


def serial_process_batch(
        input_group: List[Dict[str, Any]],
    ):
    output_group = {}
    for batch in tqdm(input_group):
        try:
            output_group[batch["custom_id"]] = call_llm_with_message(
                batch["body"]["messages"], 
                model=batch["body"]["model"]
            )
        except Exception as e:
            print(f"Error in processing batch {batch['custom_id']}: {e}")
    return output_group


def parallel_batch_to_dict(
        batch_group: List[Dict[str, Any]],
    ):
    summary_dict = {}
    for batch in batch_group:
        response_body = batch["response"]["body"]
        if response_body:
            summary_dict[batch["custom_id"]] = batch["response"]["body"]["choices"][0]["message"]["content"]
    return summary_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parallel population script.')
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--model", type=str, required=False, default="qwen2-vl-72b-instruct")
    args = parser.parse_args()

    if args.function == "batch":
        message_group, hashed_group = [], []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "": continue
                message_group.append(json.loads(line))
                hashed_group.append(hashed(line.strip()))
        batch_group = parallel_message_to_batch(message_group, hashed_group, model=args.model)
        with open(args.output, "w", encoding="utf-8") as of:
            for message in batch_group:
                of.write(json.dumps(message) + "\n")
    elif args.function == "unbatch":
        batch_group = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "": continue
                batch_group.append(json.loads(line))
        summary_dict = parallel_batch_to_dict(batch_group)
        with open(args.output, "w", encoding="utf-8") as of:
            json.dump(summary_dict, of, ensure_ascii=False, indent=4)
    elif args.function == "process":
        input_group = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "": continue
                input_group.append(json.loads(line))
        output_group = serial_process_batch(input_group)
        with open(args.output, "w", encoding="utf-8") as of:
            json.dump(output_group, of, ensure_ascii=False, indent=4)
    else:
        raise NotImplementedError(f"Function {args.function} is not supported.")
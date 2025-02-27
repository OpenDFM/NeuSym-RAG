#coding=utf8
import json, hashlib, sys, os, tiktoken
from tqdm import tqdm
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.functions.common_functions import convert_to_message, call_llm_with_message, truncate_tokens


def hashed(stringified_message: str) -> str:
    return hashlib.md5(stringified_message.encode("utf-8")).hexdigest()

PARALLEL_DICT = {}


def parallel_extract_or_fill(
        template: str,
        **kwargs
    ) -> str:
    """Extract the message to a batch file, or Fill the response of the message.

    @args:
        template (str): the template of the message

    @returns:
        str: the response of the message
    """
    stringified_message = json.dumps(convert_to_message(truncate_tokens(template), **kwargs), separators=(",", ":"))
    hashed_message = hashed(stringified_message.strip())
    parallel = kwargs.get("parallel")
    parallel_dict = {}
    if parallel.get("fill"):
        if PARALLEL_DICT.get(parallel["fill"], {}) == {}:
            PARALLEL_DICT[parallel["fill"]] = json.load(open(parallel["fill"], "r", encoding='utf-8'))
        parallel_dict = PARALLEL_DICT[parallel["fill"]]
        if parallel_dict.get(hashed_message):
            return parallel_dict[hashed_message]
        print(f"Message {hashed_message} not found in the parallel file.")
    if parallel.get("extract"):
        os.makedirs(os.path.dirname(parallel["extract"]), exist_ok=True)
        with open(parallel["extract"], "a", encoding='utf-8') as f:
            f.write(stringified_message + "\n")
        return ""
    return "NO SUMMARY"

def parallel_message_to_batch(
        input_path: str,
        output_path: str,
        model: str = 'qwen2-vl-72b-instruct',
        max_tokens: int = 1500
    ):
    """Construct a batch inference group in OpenAI style from a list of messages.

    @args:
        input_path (str): The path to the input message file.
        output_path (str): The path to the output batch file.
        model (str, optional): LLM used for batch inference. Defaults to 'qwen2-vl-72b-instruct'.
        max_tokens (int, optional): Max tokens for the response. Defaults to 1500.

    @returns:
        batch_group (List[Dict[str, Any]]): The batch group.
    """
    
    # Read messages from input file and hash the messages
    message_group, hashed_group = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "": continue
            message_group.append(json.loads(line))
            hashed_group.append(hashed(line.strip()))
    
    # Construct the OpenAI style batch
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
    
    # Write batch to the output file
    with open(output_path, "w", encoding="utf-8") as of:
        for message in batch_group:
            of.write(json.dumps(message) + "\n")
    
    return batch_group


def serial_process_batch(
        input_path: str,
        output_path: str
    ):
    input_group = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "": continue
            input_group.append(json.loads(line))
    
    output_group = {}
    for batch in tqdm(input_group, disable=not sys.stdout.isatty()):
        try:
            output_group[batch["custom_id"]] = call_llm_with_message(
                batch["body"]["messages"], 
                model=batch["body"]["model"]
            )
        except Exception as e:
            print(f"Error in processing batch {batch['custom_id']}: {e}")
            
    with open(output_path, "w", encoding="utf-8") as of:
        json.dump(output_group, of, ensure_ascii=False, indent=4)
    
    return output_group


def parallel_batch_to_dict(
        input_path: str,
        output_path: str
    ):
    """Restore the { hash_value -> response } mapping from the batch inference output.

    @args:
        input_path (str): The path to the input result file.
        output_path (str): The path to the output mapping file.

    @returns:
        summary_dict (Dict[str, str]): The mapping.
    """
    
    # Read batch results from input file
    batch_group = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "": continue
            batch_group.append(json.loads(line))
    
    # Restore the hash_value -> response mapping
    summary_dict = {}
    for batch in batch_group:
        response_body = batch["response"]["body"]
        if response_body:
            summary_dict[batch["custom_id"]] = batch["response"]["body"]["choices"][0]["message"]["content"]
    
    # Write the mapping to the output file
    with open(output_path, "w", encoding="utf-8") as of:
        json.dump(summary_dict, of, ensure_ascii=False, indent=4)
    
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
        batch_group = parallel_message_to_batch(args.input, args.output, model=args.model)
    elif args.function == "unbatch":
        summary_dict = parallel_batch_to_dict(args.input, args.output)
    elif args.function == "process":
        output_group = serial_process_batch(args.input, args.output)
    else:
        raise NotImplementedError(f"Function {args.function} is not supported.")
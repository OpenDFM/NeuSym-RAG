#coding=utf8
import os, json, sys, logging, math, random
from typing import Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dataset')


def classify_question_type(data: Dict[str, Any]) -> str:
    """ Classify the type for each question.
    @param:
        data: dict, the data extracted from test_data.jsonl
    """
    return ','.join(sorted(data.get('tags', [])))


def sampling_dataset(dataset: str = 'airqa', sample_size: int = 300, test_data: str = 'test_data.jsonl', output_file: str = None, random_seed: int = 2024):
    """ Sample the dataset for testing purposes.
    @param:
        dataset: str, the dataset name.
        sample_size: int, the sample size for the dataset.
        output_file: str, the output file name for the sampling .jsonl file.
    """
    if os.path.exists(test_data):
        dataset_path = test_data
    else:
        dataset_path = os.path.join(DATASET_DIR, dataset, test_data)
    with open(dataset_path, 'r', encoding='utf-8') as inf:
        data = [json.loads(line) for line in inf]
    typed_data = {}
    for d in data:
        question_type = classify_question_type(d)
        if question_type not in typed_data:
            typed_data[question_type] = []
        typed_data[question_type].append(d)
    # for different question types, sample the data in proportion
    for k, v in typed_data.items():
        logger.info(f"type = {k}, size = {len(v)}")
    typed_sample_size = {tp: math.ceil(sample_size * 1.0 * len(typed_data[tp]) / len(data)) for tp in typed_data}
    sampled_data = []
    random.seed(random_seed)
    for tp in typed_data:
        if len(typed_data[tp]) <= typed_sample_size[tp]:
            typed_sample_size[tp] = len(typed_data[tp])
            sampled_data.extend(typed_data[tp])
        else:
            sampled_data.extend(random.sample(typed_data[tp], typed_sample_size[tp]))
        logger.info(f'Sample {typed_sample_size[tp]} test data for type {tp}.')

    sample_size = len(sampled_data)
    output_path = os.path.join(DATASET_DIR, dataset, output_file) if output_file is not None else dataset_path.replace('test_data.jsonl', f'test_data_sample_{sample_size}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as of:
        for d in sampled_data:
            of.write(json.dumps(d, ensure_ascii=False) + '\n')
        logger.info(f"Sampled {sample_size} test data saved to {output_path} for dataset {dataset}.")
    return sampled_data


def split_dataset(dataset: str = 'airqa', split_size: int = 12, test_data: str = 'test_data.jsonl'):
    """ Split the test data into multiple parts for parallel experiments. The test data will be automatically renamed as xxxx_split_{i}.jsonl.
    """
    if os.path.exists(test_data):
        dataset_path = test_data
    else:
        dataset_path = os.path.join(DATASET_DIR, dataset, test_data)
    with open(dataset_path, 'r', encoding='utf-8') as inf:
        data = [json.loads(line) for line in inf if line.strip()]
    chunk_size = len(data) // split_size
    remainder = len(data) % split_size
    base_filename = os.path.basename(test_data).split('.')[0]
    for i in range(split_size):
        start = i * (chunk_size + 1) if i < remainder else i * chunk_size + remainder
        end = (i + 1) * (chunk_size + 1) if i < remainder else (i + 1) * chunk_size + remainder \
              if (i + 1) * chunk_size + remainder < len(data) else len(data)
        output_path = os.path.join(DATASET_DIR, dataset, f'{base_filename}_split_{i}.jsonl')
        with open(output_path, 'w', encoding='utf-8') as of:
            for d in data[start:end]:
               of.write(json.dumps(d, ensure_ascii=False) + '\n')
        logger.info(f"Split {i}: {min(len(data), end) - start} test data saved to {output_path} for dataset {dataset}.")
    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Dataset relevant utilities.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--function', type=str, default='sampling', choices=['sampling', 'split'], help='Function name.')
    parser.add_argument('--sample_size', type=int, default=300, help='Sample size for the dataset.')
    parser.add_argument('--split_size', type=int, default=12, help='Number of splits for the dataset.')
    parser.add_argument('--test_data', type=str, default='test_data.jsonl', help='Test data file name for splitting.')
    parser.add_argument('--output_file', type=str, help='Output file name of the sampling .jsonl file.')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed for sampling.')
    args = parser.parse_args()

    FUNCTIONS = {
        'sampling': sampling_dataset,
        'split': split_dataset
    }
    if args.function == 'sampling':
        FUNCTIONS[args.function](args.dataset, sample_size=args.sample_size, test_data=args.test_data, output_file=args.output_file, random_seed=args.random_seed)
    elif args.function == 'split':
        FUNCTIONS[args.function](args.dataset, split_size=args.split_size, test_data=args.test_data)
    else:
        raise ValueError(f"Function {args.function} not supported for dataset {args.dataset}.")
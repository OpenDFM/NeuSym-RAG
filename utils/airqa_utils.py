""" This module contains utility functions for the AIR-QA dataset.
"""

import os, sys, re, json, logging
from typing import Dict, Any, Optional
from utils.functions.common_functions import get_uuid

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


AIRQA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'dataset', 'airqa'
)


def get_airqa_paper_uuid(paper_title: str, meta: str = 'acl2024') -> str:
    """ Get the UUID of a paper in the AIR-QA dataset.
    `meta` is {conference}{year}, e.g., 'acl2024'.
    """
    # normalize the paper title
    paper = paper.strip()
    name = re.sub(r'[^a-z0-9_\-:]', '_', re.sub(r'\s+', ' ', paper_title.lower())) + '-' + meta
    return get_uuid(name, uuid_type='uuid5', uuid_namespace='dns')


def make_airqa_dataset(airqa_dir: str = AIRQA_DIR):
    output_path = os.path.join(airqa_dir, 'test_data.jsonl')
    indir = os.path.join(airqa_dir, 'examples')
    json_files = os.listdir(indir)
    count = 0
    with open(output_path, 'w', encoding='utf8') as of:
        for fp in json_files:
            fp = os.path.join(indir, fp)
            with open(fp, 'r', encoding='utf8') as inf:
                data = json.load(inf)
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1
    logger.info(f"Merge {count} AIR-QA examples into {output_path}.")
    return output_path
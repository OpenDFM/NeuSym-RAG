#coding=utf8
import os, sys
from dotenv import load_dotenv

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(dotenv_path=os.path.join(WORK_DIR, '.env'))

TMP_DIR = os.getenv('TMP_DIR', os.path.join(WORK_DIR, 'tmp'))
CACHE_DIR = os.getenv('CACHE_DIR', os.path.join(WORK_DIR, '.cache'))
DATASET_DIR = os.getenv('DATASET_DIR', os.path.join(WORK_DIR, 'data', 'dataset'))
DATABASE_DIR = os.getenv('DATABASE_DIR', os.path.join(WORK_DIR, 'data', 'database'))
VECTORSTORE_DIR = os.getenv('VECTORSTORE_DIR', os.path.join(WORK_DIR, 'data', 'vectorstore'))
GRAPH_DIR = os.getenv('GRAPH_DIR', os.path.join(WORK_DIR, 'data', 'graph'))

for dir in [TMP_DIR, CACHE_DIR, DATASET_DIR, DATABASE_DIR, VECTORSTORE_DIR, GRAPH_DIR]:
    if not os.path.exists(dir) or not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def get_graphrag_root(dataset: str) -> str:
    GRAPHRAG_ROOT = os.path.join(GRAPH_DIR, dataset)
    os.makedirs(GRAPHRAG_ROOT, exist_ok=True)
    return GRAPHRAG_ROOT


DATASET2DATABASE = {
    "airqa": "ai_research",
    "m3sciqa": "emnlp_papers",
    "scidqa": "openreview_papers"
}

DATABASE2DATASET = {v: k for k, v in DATASET2DATABASE.items()}
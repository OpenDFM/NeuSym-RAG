#coding=utf8
import os, sys, logging, json, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from annotation.explorer import BaseExplorer, SingleExplorer

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


uuid_file = os.path.join("data", "dataset", "airqa", "used_uuids_100.json")
with open(uuid_file, "r", encoding="utf-8") as f:
    used_uuids = json.load(f)
with open("./annotation/test_annotation.txt", "w", encoding="utf-8") as of:
    for i in range(1):
        for _ in range(100):
            pid = random.choice(used_uuids)
            try:
                explorer: BaseExplorer = SingleExplorer(pid)
                break
            except Exception as e:
                continue
        else:
            logger.info("Failed to create an explorer after 100 attemps")
            continue
        try:
            question, answer, reasoning_steps = explorer.explore()
        except Exception as e:
            logger.info(f"Failed to explore the paper {pid}. {str(e)}")
            continue
        
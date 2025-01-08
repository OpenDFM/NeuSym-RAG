#coding=utf8
import os, sys, logging, json, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from annotation.explorer import BaseExplorer, SingleExplorer
from utils.airqa_utils import generate_airqa_example_template

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

DEFAULT_LLM_MODEL = 'gpt-4o' # may be changed to other open-source models
DEFAULT_TEMPERATURE = 0.7

uuid_file = os.path.join("data", "dataset", "airqa", "used_uuids_100.json")
with open(uuid_file, "r", encoding="utf-8") as f:
    used_uuids = json.load(f)
with open("./annotation/test_annotation.txt", "w", encoding="utf-8") as of:
    for i in range(1):
        for _ in range(100):
            pid = random.choice(used_uuids) # paper uuid
            try:
                explorer: BaseExplorer = SingleExplorer(pid=pid, model=DEFAULT_LLM_MODEL, temperature=DEFAULT_TEMPERATURE)
                break
            except Exception as e:
                continue
        else:
            logger.info("Failed to create an explorer after 100 attemps")
            continue
        try:
            question, answer, reasoning_steps, tags = explorer.explore()
        except Exception as e:
            logger.info(f"Failed to explore the paper {pid}. {str(e)}")
            continue
        # question id
        qid = generate_airqa_example_template(
            question = question,
            answer_format = None,
            tags = tags,
            anchor_pdf = [pid],
            reference_pdf = [],
            conference = [],
            reasoning_steps = reasoning_steps,
            evaluator = None,
            state = None,
            annotator = DEFAULT_LLM_MODEL
        )
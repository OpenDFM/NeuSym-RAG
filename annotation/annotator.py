#coding=utf8
import os, sys, logging, re, json, random
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.airqa_utils import generate_airqa_example_template, check_airqa_examples
from utils.functions.common_functions import call_llm


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
DEFAULT_TEMPERATURE = 0.0

uuid_file = os.path.join("data", "dataset", "airqa", "used_uuids_100.json")
processed_dir = os.path.join("data", "dataset", "airqa", "processed_data")
metadata_dir = os.path.join("data", "dataset", "airqa", "metadata")


def section_partition(section_data: List[Dict[str, Any]]) -> List[str]:
    partitions = [""]
    for data in section_data:
        if data["title"].split(' ')[0].isdigit():
            partitions.append("")
        if data["title"][0].isdigit():
            partitions[-1] += f"{data['title']}\n{data['text']}\n\n"
    return partitions[1:]

def _annotate_with_llm(
        template: str, 
        model: str =  DEFAULT_LLM_MODEL, 
        temperature: float = DEFAULT_TEMPERATURE
    ) -> Any:
    llm_output = call_llm(template=template, model=model, temperature=temperature)
    logger.info(f"LLM output: \n{llm_output}\n")
    pattern = r"```(txt)?\s*\[Question\]:\s*(.*?)\s*\[Answer\]:\s*(.*?)```"
    matched = re.findall(pattern, llm_output, re.DOTALL)
    if len(matched) == 0:
        logger.info(f"Failed to match a question and answer pair.")
        return None
    question, answer = matched[-1][1].strip(), matched[-1][2].strip()
    return question, answer


def single_single_text(pdf_data: Dict[str, Any]) -> Any:
    template = """You are an intelligent annotation system who is expert in posing questions. 

You will be given a section from an AI research paper, and your task is to generate a question based on the content of the section. Your output should be in the following format:
```txt
[Question]: Your question here.
[Answer]: Your answer here.
```
Notice that:
1. Remember to wrap the question and answer with triple backticks.
2. Don't include the answer in the question.
3. Your question should be as objective as possible.
4. Your answer should be concise and clear, and should use raw context if possible.

[Context]:
{context}

Let's think step-by-step, and then provide the final question and answer."""

    section_data = pdf_data["info_from_mineru"]["TOC"]
    section_data = section_partition(section_data)
    context = f"```txt\n{random.choice(section_data).strip()}\n```"
    question, answer = _annotate_with_llm(template=template.format(context=context))
    logger.info(f"Question: {question}\nAnswer: {answer}\n")
    return question, answer


def single_single_table(pdf_data: Dict[str, Any]) -> Any:
    template = """You are an intelligent annotation system who is expert in posing questions. 

You will be given a section from an AI research paper, and your task is to generate a question based on the content in HTML format and caption of the table. Your output should be in the following format:
```txt
[Question]: Your question here.
[Answer]: Your answer here.
```
Notice that:
1. Remember to wrap the question and answer with triple backticks.
2. Don't include the answer in the question.
3. Your question should be as objective as possible.
4. Your answer should be concise and clear.
    4.1 If your answer can be just a float or integer, just provide the number.
    4.2 If your question can be presented in the form of a true-or-false statement, do so and provide the answer as `True` or `False`.
5. Try not to include the word `table` in your question.
6. Try using the numerical values in the table to ask questions, such as comparing, calculating differences, etc.

[Context]:
```txt
{context}
```

Let's think step-by-step, and then provide the final question and answer."""

    table_data = pdf_data["info_from_mineru"]["tables"]
    table_data = random.choice(table_data)
    context = f"""
Table caption: {table_data['table_caption']}
Table content in HTML format:
```html
{table_data['table_html']}
```
"""
    question, answer = _annotate_with_llm(template=template.format(context=context))
    logger.info(f"Question: {question}\nAnswer: {answer}\n")
    return question, answer


def single_multiple_single_part(pdf_data: Dict[str, Any]) -> Any:
    template = """You are an intelligent annotation system who is expert in posing questions. 

You will be given a section from an AI research paper, and your task is to generate a question based on the content of the section. Your output should be in the following format:
```txt
[Question]: Your question here.
[Answer]: Your answer here.
```
Notice that:
1. Remember to wrap the question and answer with triple backticks.
2. Don't include the answer in the question.
3. Your problem should be as objective as possible.
4. Your question should be concise and clear, and should use raw context if possible.
    4.1 If your answer can be just a float or integer, just provide the number.
    4.2 If your question can be presented in the form of a true-or-false statement, do so and provide the answer as `True` or `False`.
5. Try to pose a question with the text of the section, then pose another question with the text of the subsection. Better make the second question relyng on the first. Note that you should combine the two question into one complete question, and the two answers into one in Python List format, e.g. [answer1, answer2].
6. If there are no subsection, return "No Subsection."

[Context]:
{context}

Let's think step-by-step, and then provide the final question and answer."""

    section_data = pdf_data["info_from_mineru"]["TOC"]
    section_data = section_partition(section_data)
    context = f"```txt\n{random.choice(section_data).strip()}\n```"
    question, answer = _annotate_with_llm(template=template.format(context=context))
    logger.info(f"Question: {question}\nAnswer: {answer}\n")
    return question, answer


def single_multiple_cross_part(pdf_data: Dict[str, Any]) -> Any:
    template = """You are an intelligent annotation system who is expert in posing questions. 

You will be given a section from an AI research paper, and your task is to generate a question based on the content of the section. Your output should be in the following format:
```txt
[Question]: Your question here.
[Answer]: Your answer here.
```
Notice that:
1. Remember to wrap the question and answer with triple backticks.
2. Don't include the answer in the question.
3. Your problem should be as objective as possible.
4. Your question should be concise and clear, and should use raw context if possible.
    4.1 If your answer can be just a float or integer, just provide the number.
    4.2 If your question can be presented in the form of a true-or-false statement, do so and provide the answer as `True` or `False`.
5. Try to pose a question with the text of the first section, then pose another question with the text of the second section. Better make the second question relyng on the first. Note that you should combine the two question into one complete question, and the two answers into one in Python List format, e.g. [answer1, answer2].

[Context]:
{context}

Let's think step-by-step, and then provide the final question and answer."""

    section_data = pdf_data["info_from_mineru"]["TOC"]
    section_data = section_partition(section_data)
    indexs = sorted(random.sample(list(range(0, len(section_data))), 2))
    section_data = [section_data[index] for index in indexs]
    context = f"""First Section:
```txt
{section_data[0].strip()}
```

Second Section:
```txt
{section_data[1].strip()}
```"""
    question, answer = _annotate_with_llm(template=template.format(context=context))
    logger.info(f"Question: {question}\nAnswer: {answer}\n")
    return question, answer


with open(uuid_file, "r", encoding="utf-8") as f:
    used_uuids = json.load(f)
pid = random.choice(used_uuids)
pid = "cd91a901-384e-597b-bb20-2d4f9fa0c4f9"
logger.info(f"Randomly selected paper ID: {pid}")
with open(os.path.join(processed_dir, f"{pid}.json"), "r", encoding="utf-8") as f:
    pdf_data = json.load(f)
with open(os.path.join(metadata_dir, f"{pid}.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)
# single_single_text(pdf_data)
# single_single_table(pdf_data)
# single_multiple_single_part(pdf_data)
# single_multiple_cross_part(pdf_data)

# qid = generate_airqa_example_template()["uuid"]
# if check_airqa_examples():
#     logger.info("Successfully generated an AIR-QA example.")
# else:
#     logger.info("Failed to generate an AIR-QA example.")
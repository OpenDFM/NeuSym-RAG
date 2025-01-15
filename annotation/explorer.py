#coding=utf8
from abc import ABC, abstractmethod
import os, sys, re, json, random
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
import fitz

from utils.airqa_utils import generate_airqa_example_template, check_airqa_examples
from utils.functions.common_functions import call_llm_with_pattern
from utils.functions.image_functions import get_image_message
from agents.envs.actions.view_image import ViewImage
from agents.envs.actions.observation import Observation
from annotation.explorer_prompt import EXPLORE_PROMPT, CONTEXT_PROMPT, IMAGE_PROMPT

processed_dir = os.path.join("data", "dataset", "airqa", "processed_data")
metadata_dir = os.path.join("data", "dataset", "airqa", "metadata")

def section_partition(section_data: List[Dict[str, Any]]) -> List[str]:
    VALID_SECTION_COUNT = 5
    TITLE_KEYWORDS = [
        ["introduction"], 
        ["background", "related work", "motivation"],
        ["method", "approach", "model"],
        ["experiment", "result"],
        ["conclusion", "discussion", "future work"]
    ]
    
    def classify_title(title: str) -> int:
        for i in range(VALID_SECTION_COUNT):
            for keyword in TITLE_KEYWORDS[i]:
                if keyword in title.lower():
                    return i
        return -1
    
    index_mapping = {}
    partitions = ["" for _ in range(VALID_SECTION_COUNT)]
    for data in section_data:
        index = data["title"].split(' ')[0]
        if index.isdigit():
            index = int(index)
            index_mapping[index] = classify_title(data["title"])
            index = index_mapping[index]
            if index != -1:
                partitions[index] += f"## {data['title'].strip()}\n\n{data['text'].strip()}\n\n"
        elif index.count('.') == 1:
            index = index.split('.')[0]
            if index.isdigit() and int(index) in index_mapping:
                index = index_mapping[int(index)]
                if index != -1:
                    partitions[index] += f"### {data['title'].strip()}\n\n{data['text'].strip()}\n\n"
    for i in range(VALID_SECTION_COUNT):
        if not partitions[i].startswith("##"):
            raise ValueError(f"Section {i}: main title {TITLE_KEYWORDS[i][0]} not found.")
    return partitions


class BaseExplorer(ABC):
    pid: str = None
    model: str = None
    temperature: float = None
    pdf_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    page_data: List[str] = None
    section_data: List[str] = None
    
    def __init__(self, pid: str, model: str, temperature: float):
        self.pid = pid
        self.model = model
        self.temperature = temperature
        
        pdf_data_path = os.path.join(processed_dir, f"{self.pid}.json")
        if not os.path.exists(pdf_data_path):
            raise FileNotFoundError(f"Processed Data File {pdf_data_path} not found.")
        with open(pdf_data_path, "r", encoding="utf-8") as f:
            self.pdf_data = json.load(f)
        
        metadata_path = os.path.join(metadata_dir, f"{self.pid}.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata File {metadata_path} not found.")
        with open(os.path.join(metadata_dir, f"{self.pid}.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        pdf_path = str(self.pdf_data["pdf_path"]).replace("\\", "/")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF File {pdf_path} not found.")
        self.page_data = []
        doc = fitz.open(pdf_path)
        for page_number in range(doc.page_count):
            page = doc[page_number]
            text = page.get_text()
            self.page_data.append(text)
        doc.close()
        
    def _explore_with_llm(
            self,
            template: str,
            **kwargs
        ) -> List[Any]:
        pattern = r"```(txt)?\s*\[Question\]:\s*(.*?)\s*\[Answer\]:\s*(.*?)\s*\[Reasoning Steps\]:\s*(.*?)```"
        response = call_llm_with_pattern(template, pattern, self.model, self.temperature, **kwargs)
        if not response: raise ValueError(f"Failed to Parse the Response. {response}")
        return response[1:]
    
    @abstractmethod
    def explore(self):
        pass
    
    def get_title(self):
        return self.metadata["title"]

class SingleExplorer(BaseExplorer):
    """Single Document Explorer.
    """
    def __init__(self, pid: str, model: str, temperature: float):
        super().__init__(pid=pid, model=model, temperature=temperature)
    
    def explore(self, explore_func: str = None, **kwargs) -> Any:
        explore_funcs = ["single_text", "single_table", "single_image", "single_formula"]
        if not explore_func:
            explore_func = random.choice(explore_funcs)
        assert explore_func in explore_funcs, f"Invalid Explore Function {explore_func}."
        return getattr(self, explore_func)(**kwargs)
    
    def single_text(self, **kwargs) -> Any:
        """Single-Step Paradigm: Text Modal.
        """
        context_type = kwargs.get("context", "section")
        if context_type == "section":
            content = random.choice(section_partition(self.pdf_data["info_from_mineru"]["TOC"]))
        elif context_type == "page":
            content = random.choice(self.page_data)
        else:
            raise ValueError(f"Invalid Context Type {context_type}.")
        template = EXPLORE_PROMPT[context_type] + CONTEXT_PROMPT[context_type].format(content=content)
        question, answer, reasoning_steps = self._explore_with_llm(template)
        return question, answer, reasoning_steps, ["single", "text"]

    def single_table(self, **kwargs) -> Any:
        """Single-Step Paradigm: Table Modal.
        @kwargs:
            context: bool, whether to include additional context in the prompt.
        """
        table_data = random.choice(self.pdf_data["info_from_mineru"]["tables"])
        context = CONTEXT_PROMPT["table"].format(
            caption = table_data["table_caption"],
            content = table_data["table_html"]
        )
        tags = ["single", "table"]
        if kwargs.get("context", False):
            context += "\n\n" + CONTEXT_PROMPT["page"].format(content=self.page_data[table_data["page_number"]-1])
            tags.append("text")
        template = EXPLORE_PROMPT["table"] + context
        # print(template)
        question, answer, reasoning_steps = self._explore_with_llm(template)
        return question, answer, reasoning_steps, tags
    
    def single_image(self, **kwargs) -> Any:
        """Single-Step Paradigm: Image Modal.
        @kwargs:
            context: bool, whether to include additional context in the prompt.
        """
        image_data = [
            data 
            for data in self.pdf_data["info_from_mineru"]["figures"] 
            if data["figure_caption"].lower().startswith("figure")
        ]
        image_data = random.choice(image_data)
        context = CONTEXT_PROMPT["image"].format(caption=image_data["figure_caption"])
        tags = ["single", "image"]
        if kwargs.get("context", False):
            tags.append("text")
            context = CONTEXT_PROMPT["page"].format(content=self.page_data[image_data["page_number"]-1]) + "\n\n" + context
        template = EXPLORE_PROMPT["image"] + context
        # print(f"Caption: {image_data['figure_caption']}")
        # Get the base64 image data.
        # The following code is not a good practice, it's a temporary solution.
        action = ViewImage(paper_id=self.pid, page_number=image_data["page_number"], bounding_box=image_data["figure_bbox"])
        env = gym.Env()
        env.dataset = "airqa" # fake an environment
        obs: Observation = action.execute(env)
        base64_image: str = obs.obs_content
        
        image_template = get_image_message(base64_image, IMAGE_PROMPT)
        question, answer, reasoning_steps = self._explore_with_llm(template, image=image_template)
        
        return question, answer, reasoning_steps, tags
    
    def single_formula(self, **kwargs) -> Any:
        """Single-Step Paradigm: Formula Modal.
        @kwargs:
            context: bool, whether to include additional context in the prompt.
        """
        formula_data = self.pdf_data["info_from_mineru"]["equations"]
        if not formula_data:
            raise ValueError("No Formula Data Found.")
        index = random.randint(0, len(formula_data) - 1)
        formula_data = formula_data[index]
        context = CONTEXT_PROMPT["formula"].format(
            index=index+1,
            formula=formula_data["equation_text"]
        )
        tags = ["single", "formula"]
        if kwargs.get("context", False):
            context += "\n\n" + CONTEXT_PROMPT["page"].format(content=self.page_data[formula_data["page_number"]-1])
            tags.append("text")
        template = EXPLORE_PROMPT["formula"] + context
        question, answer, reasoning_steps = self._explore_with_llm(template)
        return question, answer, reasoning_steps, tags

    # def single_multiple_single_part(self) -> Any:
    #     template = """You are an intelligent annotation system who is expert in posing questions. 

    # You will be given a section from an AI research paper, and your task is to generate a question based on the content of the section. Your output should be in the following format:
    # ```txt
    # [Question]: Your question here.
    # [Answer]: Your answer here.
    # ```
    # Notice that:
    # 1. Remember to wrap the question and answer with triple backticks.
    # 2. Don't include the answer in the question.
    # 3. Your problem should be as objective as possible.
    # 4. Your question should be concise and clear, and should use raw context if possible.
    #     4.1 If your answer can be just a float or integer, just provide the number.
    #     4.2 If your question can be presented in the form of a true-or-false statement, do so and provide the answer as `True` or `False`.
    # 5. Try to pose a question with the text of the section, then pose another question with the text of the subsection. Better make the second question relyng on the first. Note that you should combine the two question into one complete question, and the two answers into one in Python List format, e.g. [answer1, answer2].
    # 6. If there are no subsection, return "No Subsection."

    # [Context]:
    # {context}

    # Let's think step-by-step, and then provide the final question and answer."""

    #     section_data = pdf_data["info_from_mineru"]["TOC"]
    #     section_data = section_partition(section_data)
    #     context = f"```txt\n{random.choice(section_data).strip()}\n```"
    #     question, answer = _annotate_with_llm(template=template.format(context=context))
    #     # logger.info(f"Question: {question}\nAnswer: {answer}\n")
    #     return question, answer


# def single_multiple_cross_part(pdf_data: Dict[str, Any]) -> Any:
#     template = """You are an intelligent annotation system who is expert in posing questions. 

# You will be given a section from an AI research paper, and your task is to generate a question based on the content of the section. Your output should be in the following format:
# ```txt
# [Question]: Your question here.
# [Answer]: Your answer here.
# ```
# Notice that:
# 1. Remember to wrap the question and answer with triple backticks.
# 2. Don't include the answer in the question.
# 3. Your problem should be as objective as possible.
# 4. Your question should be concise and clear, and should use raw context if possible.
#     4.1 If your answer can be just a float or integer, just provide the number.
#     4.2 If your question can be presented in the form of a true-or-false statement, do so and provide the answer as `True` or `False`.
# 5. Try to pose a question with the text of the first section, then pose another question with the text of the second section. Better make the second question relyng on the first. Note that you should combine the two question into one complete question, and the two answers into one in Python List format, e.g. [answer1, answer2].

# [Context]:
# {context}

# Let's think step-by-step, and then provide the final question and answer."""

#     section_data = pdf_data["info_from_mineru"]["TOC"]
#     section_data = section_partition(section_data)
#     indexs = sorted(random.sample(list(range(0, len(section_data))), 2))
#     section_data = [section_data[index] for index in indexs]
#     context = f"""First Section:
# ```txt
# {section_data[0].strip()}
# ```

# Second Section:
# ```txt
# {section_data[1].strip()}
# ```"""
#     question, answer = _annotate_with_llm(template=template.format(context=context))
#     # logger.info(f"Question: {question}\nAnswer: {answer}\n")
#     return question, answer
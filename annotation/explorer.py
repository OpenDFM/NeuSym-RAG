#coding=utf8
from abc import ABC, abstractmethod
import os, sys, re, json, random
from typing import List, Dict, Any
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
import fitz

from utils.airqa_utils import generate_airqa_example_template, check_airqa_examples
from utils.functions.common_functions import call_llm_with_pattern, call_llm, convert_to_message, call_llm_with_message
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
    model: str = None
    temperature: float = None
    
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        
    def _explore_with_llm(
            self,
            template: str,
            **kwargs
        ) -> List[Any]:
        messages = convert_to_message(template, **kwargs)
        response = call_llm_with_message(messages, model=self.model, temperature=self.temperature)
        messages.append({"role": "assistant", "content": response})
        return messages
    
    @abstractmethod
    def explore(self, **kwargs) -> Any:
        pass

class SingleExplorer(BaseExplorer):
    """Single Document Explorer.
    """
    exp_type: str = "single"
    pid: str = None
    pdf_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    page_data: List[str] = None
    
    def __init__(self, pid: str, model: str, temperature: float):
        super().__init__(model=model, temperature=temperature)
        self.pid = pid
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
    
    def get_title(self) -> str:
        return self.metadata["title"]
    
    def get_conference(self) -> str:
        return str(self.metadata["conference"]).lower() + str(self.metadata["year"])
    
    def get_titles(self) -> List[str]:
        return [self.metadata["title"]]
    
    def explore(self, **kwargs) -> Any:
        # explore_funcs = ["single_text", "single_table", "single_image", "single_formula", "multiple_section_subsection", "multiple_section_section"]
        explore_funcs = ["single_text", "single_table", "single_image", "multiple_section_subsection", "multiple_section_section"]
        explore_func = kwargs.get("explore_func", None)
        if not explore_func: explore_func=random.choice(explore_funcs)
        assert explore_func in explore_funcs, f"Invalid Explore Function {explore_func}."
        return getattr(self, explore_func)(**kwargs)
    
    def single_text(self, **kwargs) -> Any:
        """Single-Step Paradigm: Text Modal.
        @kwargs:
            context: str, default "section", the context type, either "section" or "page".
        """
        context_type = kwargs.get("context", "section")
        if context_type == "section":
            content = random.choice(section_partition(self.pdf_data["info_from_mineru"]["TOC"]))
        elif context_type == "page":
            content = random.choice(self.page_data)
        else:
            raise ValueError(f"Invalid Context Type {context_type}.")
        template = EXPLORE_PROMPT[self.exp_type][context_type] + CONTEXT_PROMPT[self.exp_type][context_type].format(content=content)
        return self._explore_with_llm(template), [self.exp_type, "text"]

    def single_table(self, **kwargs) -> Any:
        """Single-Step Paradigm: Table Modal.
        @kwargs:
            context: bool, default True, whether to include additional context in the prompt.
        """
        table_data = random.choice(self.pdf_data["info_from_mineru"]["tables"])
        context = CONTEXT_PROMPT[self.exp_type]["table"].format(
            caption = table_data["table_caption"],
            content = table_data["table_html"]
        )
        tags = [self.exp_type, "table"]
        if kwargs.get("context", True):
            context += "\n\n" + CONTEXT_PROMPT[self.exp_type]["page"].format(content=self.page_data[table_data["page_number"]-1])
            tags.append("text")
        template = EXPLORE_PROMPT[self.exp_type]["table"] + context
        return self._explore_with_llm(template), tags
    
    def single_image(self, **kwargs) -> Any:
        """Single-Step Paradigm: Image Modal.
        @kwargs:
            context: bool, default True, whether to include additional context in the prompt.
        """
        image_data = [
            data 
            for data in self.pdf_data["info_from_mineru"]["figures"] 
            if data["figure_caption"].lower().startswith("figure")
        ]
        image_data = random.choice(image_data)
        context = CONTEXT_PROMPT[self.exp_type]["image"].format(caption=image_data["figure_caption"])
        tags = [self.exp_type, "image"]
        if kwargs.get("context", True):
            tags.append("text")
            context = CONTEXT_PROMPT[self.exp_type]["page"].format(content=self.page_data[image_data["page_number"]-1]) + "\n\n" + context
        template = EXPLORE_PROMPT[self.exp_type]["image"] + context
        # Get the base64 image data.
        # The following code is not a good practice, it's a temporary solution.
        def transfer_bbox(bbox: List[Any]) -> List[Any]: # original airqa-100 bbox got some problem
            return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        action = ViewImage(paper_id=self.pid, page_number=image_data["page_number"], bounding_box=transfer_bbox(image_data["figure_bbox"]))
        env = gym.Env()
        env.dataset = "airqa" # fake an environment
        obs: Observation = action.execute(env)
        base64_image: str = obs.obs_content
        
        image_template = get_image_message(base64_image, IMAGE_PROMPT[self.exp_type])
        return self._explore_with_llm(template, image=image_template), tags
    
    # The question posed using formula is not satisfying, due to the wrong index.
    def single_formula(self, **kwargs) -> Any:
        """Single-Step Paradigm: Formula Modal.
        @kwargs:
            context: bool, default True, whether to include additional context in the prompt.
        """
        formula_data = self.pdf_data["info_from_mineru"]["equations"]
        if not formula_data:
            raise ValueError("No Formula Data Found.")
        index = random.randint(0, len(formula_data) - 1)
        formula_data = formula_data[index]
        context = CONTEXT_PROMPT[self.exp_type]["formula"].format(
            index=index+1,
            formula=formula_data["equation_text"]
        )
        tags = [self.exp_type, "formula"]
        if kwargs.get("context", True):
            context += "\n\n" + CONTEXT_PROMPT[self.exp_type]["page"].format(content=self.page_data[formula_data["page_number"]-1])
            tags.append("text")
        template = EXPLORE_PROMPT[self.exp_type]["formula"] + context
        return self._explore_with_llm(template), tags

    def multiple_section_subsection(self, **kwargs) -> Any:
        """Multiple-Step Paradigm: Section-Subsection Modal.
        """
        content = random.choice(section_partition(self.pdf_data["info_from_mineru"]["TOC"]))
        template = EXPLORE_PROMPT[self.exp_type]["sec_sub"] + CONTEXT_PROMPT[self.exp_type]["sec_sub"].format(content=content)
        return self._explore_with_llm(template), [self.exp_type, "text"]

    def multiple_section_section(self, **kwargs) -> Any:
        """Multiple-Step Paradigm: Section-Section Modal.
        """
        section_data = section_partition(self.pdf_data["info_from_mineru"]["TOC"])
        indexs = sorted(random.sample(list(range(0, len(section_data))), 2))
        section_data = [section_data[index].strip() for index in indexs]
        context = CONTEXT_PROMPT[self.exp_type]["sec_sec"].format(content0=section_data[0], content1=section_data[1])
        template = EXPLORE_PROMPT[self.exp_type]["sec_sec"] + context
        return self._explore_with_llm(template), [self.exp_type, "text"]

class MultipleExplorer(BaseExplorer):
    """Multiple Document Explorer.
    """
    exp_type: str = "multiple"
    pid: List[str] = None
    subexplorers: List[SingleExplorer] = []
    def __init__(self, pid: List[str], model: str, temperature: float):
        super().__init__(model=model, temperature=temperature)
        self.pid = pid
        self.subexplorers = [SingleExplorer(pid=_pid, model=model, temperature=temperature) for _pid in pid]
    
    def get_conference(self):
        return None
    
    def get_titles(self) -> List[str]:
        return [explorer.get_title() for explorer in self.subexplorers]
    
    def explore(self, **kwargs) -> Any:
        explore_funcs = ["multiple_text", "multiple_table", "multiple_image"]
        explore_func = kwargs.get("explore_func", None)
        if not explore_func: explore_func=random.choice(explore_funcs)
        assert explore_func in explore_funcs, f"Invalid Explore Function {explore_func}."
        return getattr(self, explore_func)(**kwargs)
    
    def multiple_text(self, **kwargs) -> Any:
        """Comparison Paradigm: Text Modal.
        """
        context_type = kwargs.get("context", "section")
        if context_type == "section":
            part = random.randint(0, 5)
            content = [section_partition(explorer.pdf_data["info_from_mineru"]["TOC"])[part] for explorer in self.subexplorers]
        elif context_type == "page":
            content = [random.choice(explorer.page_data) for explorer in self.subexplorers]
        else:
            raise ValueError(f"Invalid Context Type {context_type}.")
        context = "\n\n".join([CONTEXT_PROMPT[self.exp_type][context_type].format(index=i+1, content=content[i]) for i in range(len(content))])
        template = EXPLORE_PROMPT[self.exp_type][context_type] + context
        return self._explore_with_llm(template), [self.exp_type, "text"]
    
    def multiple_table(self, **kwargs) -> Any:
        """Comparison Paradigm: Table Modal.
        @kwargs:
            context: bool, default True, whether to include additional context in the prompt.
        """
        table_data = [random.choice(explorer.pdf_data["info_from_mineru"]["tables"]) for explorer in self.subexplorers]
        page_data = [random.choice(explorer.page_data) for explorer in self.subexplorers]
        context = ""
        for i in range(len(table_data)):
            context += CONTEXT_PROMPT[self.exp_type]["table"].format(
                index = i+1,
                caption = table_data[i]["table_caption"],
                content = table_data[i]["table_html"]
            ) + "\n\n"
            if kwargs.get("context", True):
                context += CONTEXT_PROMPT[self.exp_type]["page"].format(
                    index = i+1,
                    content=page_data[i][table_data[i]["page_number"]-1]
                ) + "\n\n"
        tags = [self.exp_type, "table"]
        if kwargs.get("context", True):
            tags.append("text")
        template = EXPLORE_PROMPT[self.exp_type]["table"] + context
        return self._explore_with_llm(template), tags
    
    def multiple_image(self, **kwargs) -> Any:
        """Comparison Paradigm: Image Modal.
        @kwargs:
            context: bool, default True, whether to include additional context in the prompt.
        """
        image_data = [
            random.choice([
                data for data in explorer.pdf_data["info_from_mineru"]["figures"]
                if data["figure_caption"].lower().startswith("figure")
            ]) for explorer in self.subexplorers
        ]
        page_data = [random.choice(explorer.page_data) for explorer in self.subexplorers]
        context = ""
        for i in range(len(image_data)):
            context += CONTEXT_PROMPT[self.exp_type]["image"].format(
                index = i+1,
                caption = image_data[i]["figure_caption"],
            ) + "\n\n"
            if kwargs.get("context", True):
                context += CONTEXT_PROMPT[self.exp_type]["page"].format(
                    index = i+1,
                    content=page_data[i][image_data[i]["page_number"]-1]
                ) + "\n\n"
        tags = [self.exp_type, "image"]
        if kwargs.get("context", True):
            tags.append("text")
        template = EXPLORE_PROMPT[self.exp_type]["image"] + context
        
        def transfer_bbox(bbox: List[Any]) -> List[Any]:
            return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        
        env = gym.Env()
        env.dataset = "airqa" # fake an environment
        image_template = []
        for i in range(len(image_data)):
            action = ViewImage(paper_id=self.pid[i], page_number=image_data[i]["page_number"], bounding_box=transfer_bbox(image_data[i]["figure_bbox"]))
            obs: Observation = action.execute(env)
            base64_image: str = obs.obs_content
            image_template.append(get_image_message(base64_image, IMAGE_PROMPT[self.exp_type].format(index=i+1)))
        
        return self._explore_with_llm(template, image=image_template), tags

class ComprehensiveExplorer(SingleExplorer):
    """Comprehensive Document Explorer.
    """
    exp_type: str = "comprehensive"
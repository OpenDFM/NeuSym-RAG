#coding=utf8

import os, openai, uuid, re
from typing import List, Dict, Union, Optional, Any, Iterable
import fitz  # PyMuPDF
from utils.functions.common_functions import call_llm, get_uuid


def get_pdf_page_text(
        pdf_path: str,
        generate_uuid: bool = True,
        uuid_type: str = 'uuid5',
        uuid_namespace: str = 'dns',
        normalize_blank: bool = True
    ) -> Dict[str, Union[str, List[str]]]:
    """ Extract the content of each page from the PDF file.
    @args:
        pdf_path: str, the path to the PDF file.
        generate_uuid: bool, whether to generate the UUID for each page, default to False.
        uuid_type: str, chosen from uuid3, uuid4, uuid5, default to uuid5.
        uuid_namespace: str, chosen from dns, url, oid, x500, default to dns.
        normalize_blank: bool, whether to normalize the blank lines, default to True.
    @return:
        output: Dict[str, Union[str, List[str]]], the output dictionary containing the following keys:
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
            - page_contents: List[str], the list of strings, each string represents the content of each page.
            - page_uuids: List[str], the list of UUIDs for each page if generate_uuid is True.
    """
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    pdf_id = None
    if generate_uuid:
        pdf_id = get_uuid(name=pdf_path, uuid_type=uuid_type, uuid_namespace=uuid_namespace)
    page_contents, page_uuids = [], []
    for page_number in range(doc.page_count):
        page = doc[page_number]
        text = page.get_text()
        
        if normalize_blank: # replace multiple blank lines with one
            text = re.sub(r'\n+', '\n', text)
        page_contents.append(text)
        
        if generate_uuid:
            page_uuids.append(get_uuid(name=text, uuid_type=uuid_type, uuid_namespace=uuid_namespace))
    output = {"pdf_id": pdf_id, "pdf_name": file_name, "pdf_path": pdf_path, "page_contents": page_contents, "page_uuids": page_uuids}
    return output


def get_text_summary(
        content: Dict[str, Any],
        key: str = 'page_contents',
        max_length: int = 50,
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
    """ Get the content summary by extracting the first `num_sentences` sentences from the content.
    @args:
        content: str, the content to be summarized.
        max_length: int, the maximum length of the summary, default to 50.
    @return:
        summary: str, the content summary.
    """
    prompt_template = """You are an intelligent assistant who is expert at summarizing long text content.
    
The content is as follows:
{text}
Please directly return the summary without any extra information or formatting. And you should summarize it in no more than {max_length} words. Here is your summary:
"""
    texts = content[key]
    if type(texts) == str:
        texts = [texts]
    
    summary = []
    for text in texts:
        if len(text) > max_length:
            summary.append(call_llm(prompt_template.format(text=text, max_length=max_length), model=model, top_p=top_p, temperature=temperature))
        else:
            summary.append(text)
    return {'text_summary': summary if type(content[key]) == list else summary[0]}
#coding=utf8
import json, uuid, sys, os, re, logging
from typing import List, Union, Optional, Tuple, Any, Dict
import fitz, pymupdf # PyMuPDF
import PyPDF2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage, LTRect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.functions.common_functions import get_uuid, call_llm, call_llm_with_message
from utils.functions.pdf_functions import get_pdf_page_text, load_json_from_processed_data, get_table_summary, get_text_summary
from utils.functions.image_functions import get_image_summary
from utils.functions.ai_research_metadata import AIRQA_DIR, get_airqa_paper_uuid
from fuzzywuzzy.fuzz import partial_ratio


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_ai_research_pdf_data(
        pdf_path: str, 
        processed_data_folder: str = 'data/dataset/airqa/processed_data',
        TOC_threshold: float = 0.9
    ) -> Dict[str, Any]:
    """ Load the parsed JSON data from a PDF file. See `utils.function.pdf_functions.parse_pdf` for more details.
        Output (pdf_data)
    """
    return load_json_from_processed_data(pdf_path=pdf_path, processed_data_folder=processed_data_folder, TOC_threshold=TOC_threshold)


def get_ai_research_page_info(
        metadata: Dict[str, Any], 
        max_length: int = 50,
        model: str = 'gpt-4o', 
        temperature: float = 0.7, 
        top_p: float = 0.95
    ) -> Dict[str, Union[str, List[str]]]:
    """ Output (page_data):
        Dict[str, Union[str, List[str]]], the output dictionary containing the following keys:
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
            - page_contents: List[str], the list of strings, each string represents the content of each page.
            - page_summaries: List[str], the list of text summaries for each page.
            - page_uuids: List[str], the list of UUIDs for each page.
    """
    pdf_path = metadata["pdf_path"]
    pdf_name = metadata["uuid"]
    num_pages = metadata["num_pages"]
    page_info = get_pdf_page_text(pdf_path)
    page_info["page_summaries"] = get_text_summary(
        content={"page_contents": page_info["page_contents"]},
        max_length=max_length,
        model=model,
        temperature=temperature,
        top_p=top_p
    ).get("text_summary", [])
    page_info["page_uuids"] = [
        get_uuid(name=f"{pdf_name}_page_{page_number}") for page_number in range(1, num_pages + 1)
    ]
    return page_info


def get_ai_research_per_page_chunk_info(
        metadata: Dict[str, Any], 
        page_data: Dict[str, Any], 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[List[Dict[str, str]]]:
    """ Output (chunk_data):
        [
            [{'uuid': uuid1, 'text': text1 of page 1}, {'uuid': uuid2, 'text': text2 of page 1}, ...],
            [{'uuid': uuid1, 'text': text1 of page 2}, {'uuid': uuid2, 'text': text2 of page 2}, ...],
            ...
        ]
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    results = []
    pdf_name = metadata["uuid"]
    
    for page_idx, page_content in enumerate(page_data['page_contents'], start=1):
        page_chunks = text_splitter.create_documents([page_content])
        page_results = []
        
        for chunk_idx, chunk in enumerate(page_chunks, start=1):
            chunk_uuid = get_uuid(name=f"{pdf_name}_page_{page_idx}_chunk_{chunk_idx}")
            page_results.append({'uuid': chunk_uuid, 'text': chunk.page_content})
        
        results.append(page_results)
    
    return results


def get_ai_research_section_info(
        metadata: Dict[str, Any], 
        pdf_data: Dict[str, Any],
        max_length: int = 50,
        model: str = 'gpt-4o', 
        temperature: float = 0.7, 
        top_p: float = 0.95
    ) -> List[Dict[str, Union[str, List[int]]]]:
    """ Output (section_data):
        [
            {'uuid': uuid1, 'title': introduction, 'text': content of section1, 'summary': summary of section1, 'page_numbers':[1,2]}, 
            {'uuid': uuid2, 'title': background, 'text': content of section2, 'summary': summary of section2,'page_numbers':[3]}
            ...
        ]
    """
    pdf_name = metadata["uuid"]
    sections=[]
    for idx, section_data in enumerate(pdf_data["info_from_mineru"]["TOC"]):
        title= section_data["title"]
        section_text= section_data["text"]
        section_summary = get_text_summary(
            content={"section_text": section_text},
            key="section_text",
            max_length=max_length,
            model=model,
            temperature=temperature,
            top_p=top_p
        ).get("text_summary", "")
        sections.append({
            'uuid': get_uuid(name=f"{pdf_name}_section_{idx}"),
            'title': title,
            'text': section_text,
            'summary': section_summary,
            'page_numbers': section_data["page_numbers"]
        })

    return sections


def get_ai_research_per_page_table_info(
        metadata: str, 
        pdf_data: Dict[str, Any],
        max_length: int = 50,
        model: str = 'gpt-4o', 
        temperature: float = 0.7, 
        top_p: float = 0.95
    ) -> List[List[Dict[str, Any]]]:
    """ Output (table_data):
        [ [ {'uuid': uuid1 of page1, 'table_content': table_html ,'table_caption': table caption1, 'bbox': bbox1, 'table_summary':table_summary}, {...} ], [ {...} ] ... ]
    """
    pdf_data_mineru = pdf_data["info_from_mineru"]
    pdf_path = metadata["pdf_path"]
    pdf_name = metadata["uuid"]
    num_pages = metadata["num_pages"]
    results = []

    # Group tables by page number
    tables_by_page = {}
    for table in pdf_data_mineru.get("tables", []):
        page_number = table.get("page_number", 0)
        if page_number not in tables_by_page:
            tables_by_page[page_number] = []
        tables_by_page[page_number].append(table)

    for page_num in range(1, num_pages + 1):
        if page_num in tables_by_page:
            page_tables = tables_by_page[page_num]
            page_result = []

            for ordinal, table in enumerate(page_tables, start=1):
                uuid = get_uuid(name=f"{pdf_name}_page_{page_num}_table_{ordinal}")
                table_summary = get_table_summary(
                    table=table,
                    max_length=max_length,
                    model=model,
                    temperature=temperature,
                    top_p=top_p
                )

                table_info = {
                    "uuid": uuid,
                    "table_content": table["table_html"],
                    "table_caption": table["table_caption"],
                    "bbox": table["table_bbox"],
                    "table_summary": table_summary
                }
                page_result.append(table_info)

            results.append(page_result)  # Append tables for this page
        else:
            results.append([])  # Append empty list if no tables on this page

    return results


def get_ai_research_per_page_image_info(
        metadata: Dict[str, Any], 
        pdf_data: Dict[str, Any],
        max_length: int = 50,
        model: str = 'gpt-4o', 
        temperature: float = 0.7, 
        top_p: float = 0.95
    ) -> List[List[Dict[str, Any]]]:
    """ Output (image_data):
        [ [ {'uuid': uuid, 'image_caption': image_caption, 'image_summary': immage_summary, 'bbox': bbox1}, {...} ], [ {...} ] ... ]
    """
    pdf_data_mineru = pdf_data["info_from_mineru"]
    pdf_name = metadata["uuid"]
    num_pages = metadata["num_pages"]
    
    images_by_page = {}
    for image in pdf_data_mineru.get("figures", []):
        page_number = image.get("page_number", 0)
        if page_number not in images_by_page:
            images_by_page[page_number] = []
        images_by_page[page_number].append(image)
    
    results = []
    for page_num in range(1, num_pages + 1):
        images = images_by_page.get(page_num, [])
        result = []
        for ordinal, image in enumerate(images, start=1):
            uuid = get_uuid(name=f"{pdf_name}_page_{page_num}_image_{ordinal}")
            image_summary = get_image_summary(
                image_path=image["figure_path"],
                max_length=max_length,
                model=model,
                temperature=temperature,
                top_p=top_p
            )
            image_info = {
                "uuid": uuid,
                "image_caption": image["figure_caption"],
                "image_summary": image_summary,
                "bbox": image["figure_bbox"]
            }
            result.append(image_info)
        results.append(result)
    
    return results


def get_ai_research_per_page_equation_info(
        metadata: Dict[str, Any], 
        pdf_data: Dict[str, Any]
    ) -> List[List[Dict[str, str]]]:
    """ Output (equation_data):
        [
            [{'uuid': uuid, 'text': equation1 of page 1}, {'uuid': uuid, 'text': equation2 of page 1}, ...],
            [{'uuid': uuid, 'text': equation1 of page 2}, {'uuid': uuid, 'text': equation2 of page 2}, ...],
            ...
        ]
    """
    results = []

    pdf_data_mineru = pdf_data["info_from_mineru"]
    pdf_name = metadata["uuid"]
    num_pages = metadata["num_pages"]
    
    equations_by_page = {}
    for equation in pdf_data_mineru.get("equations", []):
        page_number = equation.get("page_number", 0)
        if page_number not in equations_by_page:
            equations_by_page[page_number] = []
        equations_by_page[page_number].append(equation)

    for page_num in range(1, num_pages + 1):
        result = []
        equations = equations_by_page.get(page_num, [])
        for ordinal, equation in enumerate(equations, start=1):
            equation_text = equation.get("equation_text", "")
            if equation_text != "":
                uuid = get_uuid(name=f"{pdf_name}_page_{page_num}_equation_{ordinal}")
                result.append({'uuid': uuid, 'text': equation_text})
        results.append(result)
    
    return results


def get_ai_research_reference_info(
        metadata: Dict[str, Any],
        pdf_data: Dict[str, Any],
        page_data: Dict[str, Any],
        threshold: float = 0.9
    ) -> List[Dict[str, str]]:
    """ Output (reference_data):
        [ {'uuid': uuid, 'text': text, 'ref_page_id': page_id}, {...} ]
    """
    results = []
    pdf_data_mineru = pdf_data["info_from_mineru"]
    pdf_name = metadata["uuid"]
    num_pages = metadata["num_pages"]
    for idx, reference in enumerate(pdf_data_mineru.get("references", []), start=1):
        uuid = get_uuid(name=f"{pdf_name}_reference_{idx}")
        text = reference.get("reference_text", "")
        page_number = -1
        for i in range(num_pages):
            if float(partial_ratio(page_data["page_contents"][i], text)) / 100.0 > threshold:
                page_number = i
        if page_number != -1:
            results.append({'uuid': uuid, 'text': text, 'ref_page_id': page_data["page_uuids"][page_number]})
    return results


def aggregate_ai_research_pages(metadata: Dict[str, Any], page_data: Dict[str, Any]) -> List[List[Any]]:
    """ Output:
        [ [ page_id, page_number, page_width, page_height, page_content, page_summary, ref_paper_id ] ]
    """
    
    pdf_path = metadata["pdf_path"]
    pdf_name = metadata["uuid"]
    num_pages = metadata["num_pages"]
    doc = fitz.open(pdf_path)
    results = []

    for page_number in range(1, num_pages + 1):
        # Get the corresponding page from the PDF
        page = doc[page_number - 1]
        page_width = page.rect.width
        page_height = page.rect.height
        page_uuid = page_data["page_uuids"][page_number - 1]
        page_content = page_data["page_contents"][page_number - 1]
        page_summary = page_data["page_summaries"][page_number - 1]

        results.append([
            page_uuid,        # page_id
            page_number,      # page_number (starts from 1)
            int(page_width),  # page_width
            int(page_height), # page_height
            page_content,     # page_content
            page_summary,     # page_summary
            pdf_name          # ref_paper_id
        ])

    doc.close()

    return results

def aggregate_ai_research_chunks(
        metadata: Dict[str, Any], 
        chunk_data: List[List[Dict[str, str]]], 
        page_data: Dict[str, Any]
    ) -> List[List[Any]]:
    """ Output:
        [ [ chunk_id, text_content, ordinal, ref_paper_id, ref_page_id ] ]
    """
    paper_id = metadata["uuid"]
    results = []

    for page_index, page_chunks in enumerate(chunk_data):
        ref_page_id = page_data["page_uuids"][page_index]
        for ordinal, chunk in enumerate(page_chunks, start=1):
            results.append([
                chunk["uuid"],       # chunk_id
                chunk["text"],       # text_content
                ordinal,             # ordinal (order within the page)
                paper_id,            # ref_paper_id
                ref_page_id          # ref_page_id
            ])

    return results

def aggregate_ai_research_tables(
        metadata: Dict[str, Any], 
        table_data: list, 
        page_data: Dict[str, Any]
    ) -> List[List[Any]]:
    """ Output:
        [ [ table_id, table_caption, table_html, table_summary, table_bbox, table_ordinal, ref_paper_id, ref_page_id ] ]
    """
    
    paper_id = metadata["uuid"]
    results = []

    for page_idx, tables_in_page in enumerate(table_data):
        if not tables_in_page:
            continue

        ref_page_id = page_data["page_uuids"][page_idx]

        for table_idx, table in enumerate(tables_in_page):
            table_id = table["uuid"]
            table_caption = table["table_caption"]
            table_html = table["table_content"]
            table_summary = table["table_summary"]
            table_bbox = table["bbox"]
            table_ordinal = table_idx 

            results.append([
                table_id,                   # table_id
                table_caption,              # table_caption
                table_html,                 # table_content
                table_summary,              # table_summary
                json.dumps(table_bbox),     # bounding_box
                table_ordinal,              # ordinal
                paper_id,                   # ref_paper_id
                ref_page_id                 # ref_page_id
            ])

    return results

def aggregate_ai_research_sections(
        metadata: Dict[str, Any], 
        section_data: List[Dict[str, Union[str, List[int]]]]
    ) -> List[List[Any]]:
    """ 
    Output:
        [ [ section_id, section_title, section_text, section_summary, ordinal, page_numbers, ref_paper_id ] ]
    """
    
    paper_id = metadata["uuid"]
    results = []

    for ordinal, section in enumerate(section_data):
        section_id = section["uuid"]
        section_title = section["title"]
        section_text = section["text"]
        section_summary = section["summary"]
        page_numbers = section["page_numbers"]

        results.append([
            section_id,     # section_id
            section_title,  # section_title
            section_text,   # section_content
            section_summary,# section_summary
            ordinal,        # ordinal
            page_numbers,   # page_numbers
            paper_id        # ref_paper_id
        ])

    return results

def aggregate_ai_research_images(
        metadata: Dict[str, Any], 
        image_data: List[List[Dict[str, Any]]], 
        page_data: Dict[str, Any]
    ) -> List[List[Any]]:
    """ Output:
        [ [ image_id, image_caption, image_summary, bounding_box, ordinal, ref_paper_id, ref_page_id ] ]
    """
    paper_id = metadata["uuid"]
    results = []

    for page_idx, images_in_page in enumerate(image_data):
        if not images_in_page:
            continue

        ref_page_id = page_data["page_uuids"][page_idx]

        for image_idx, image in enumerate(images_in_page):
            image_id = image["uuid"]
            image_caption = image["image_caption"]
            image_summary = image["image_summary"]
            bounding_box = image["bbox"]
            ordinal = image_idx

            results.append([
                image_id,                   # image_id
                image_caption,              # image_caption
                image_summary,              # image_summary
                json.dumps(bounding_box),   # bounding_box
                ordinal,                    # ordinal
                paper_id,                   # ref_paper_id
                ref_page_id                 # ref_page_id
            ])

    return results

def aggregate_ai_research_equations(
        metadata: Dict[str, Any], 
        equation_data: List[List[Dict[str, str]]], 
        page_data: Dict[str, Any]
    ) -> List[List[Any]]:
    """ Output:
        [ [ equation_id, equation_content, ordinal, ref_paper_id, ref_page_id ] ]
    """
    paper_id = metadata["uuid"]
    results = []

    for page_idx, equations_in_page in enumerate(equation_data):
        if not equations_in_page:
            continue

        ref_page_id = page_data["page_uuids"][page_idx]

        for idx, equation in enumerate(equations_in_page):
            equation_id = equation["uuid"]
            equation_content = equation["text"]
            ordinal = idx

            results.append([
                equation_id,        # equation_id
                equation_content,   # equation_content
                ordinal,            # ordinal
                paper_id,           # ref_paper_id
                ref_page_id         # ref_page_id
            ])

    return results


def aggregate_ai_research_references(
        metadata: Dict[str, Any],
        reference_data: List[Dict[str, str]],
    ) -> List[List[Any]]:
    """ Output:
        [ [ reference_id, reference_content, ordinal, ref_paper_id ] ]
    """
    paper_id = metadata["uuid"]
    results = []
    
    for ordinal, reference in enumerate(reference_data):
        reference_id = reference["uuid"]
        reference_content = reference["text"]
        ref_page_id = reference["ref_page_id"]

        results.append([
            reference_id,       # reference_id
            reference_content,  # reference_content
            ordinal,            # ordinal
            paper_id,           # ref_paper_id
            ref_page_id,        # ref_page_id
        ])
    
    return results
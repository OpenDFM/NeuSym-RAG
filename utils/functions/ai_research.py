#coding=utf8
import json, uuid, sys, os, re, logging
from typing import List, Union, Optional, Tuple, Any, Dict
import fitz, pymupdf # PyMuPDF
import PyPDF2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage, LTRect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from difflib import SequenceMatcher
from utils.functions.common_functions import get_uuid, call_llm, call_llm_with_message
from utils.functions.pdf_functions import get_pdf_page_text, load_json_from_processed_data
from utils.functions.image_functions import get_image_summary
from utils.airqa_utils import AIRQA_DIR, get_airqa_paper_uuid


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
        processed_data_folder: str = 'data/dataset/airqa/processed_data'
    ) -> Dict[str, Any]:
    """ Load the parsed JSON data from a PDF file. See `utils.function.pdf_functions.parse_pdf` for more details.
        Output (pdf_data)
    """
    return load_json_from_processed_data(pdf_path=pdf_path, processed_data_folder=processed_data_folder)


def get_ai_research_page_info(pdf_path: str) -> Dict[str, Union[str, List[str]]]:
    """ Output (page_data):
        Dict[str, Union[str, List[str]]], the output dictionary containing the following keys:
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
            - page_contents: List[str], the list of strings, each string represents the content of each page.
            - page_uuids: List[str], the list of UUIDs for each page.
    """
    return get_pdf_page_text(pdf_path)


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
        page_data: Dict[str, Any], 
        threshold: float = 0.9
    ) -> List[Dict[str, Union[str, List[int]]]]:
    """ Output (section_data):
        [
            {'uuid': uuid1, 'title': introduction, 'text': content of section1, 'page_numbers':[1,2]}, 
            {'uuid': uuid2, 'title': background, 'text': content of section2, 'page_numbers':[3]}
            ...
        ]
    """
    sections = []
    pdf_name = metadata["uuid"]
    page_contents = page_data['page_contents']

    # Fuzzy matching function, combining finding matches and returning positions
    def find_fuzzy_match_positions(content: str, target: str) -> List[Tuple[int, int]]:
        """
        Perform fuzzy matching to find positions of the target in the content.
        Returns a list of start and end positions for matches.
        """
        lines = content.splitlines()
        matches = []
        for line in lines:
            # Use SequenceMatcher to compare similarity
            similarity = SequenceMatcher(None, line.lower(), target.lower()).ratio()
            if similarity > threshold:  # Threshold for a match
                match = re.search(re.escape(line), content)
                if match:
                    matches.append((match.start(), match.end()))  # Store start and end positions
        return matches

    # Extract all level 1 bookmarks
    toc_entries = [entry for entry in pdf_data["TOC"] if entry["level"] == 1]

    for i, toc_entry in enumerate(toc_entries):
        # Get the title and page number of the current section
        start_title = toc_entry["title"]
        start_page = toc_entry["page_number"] - 1  # Adjust for zero-indexed
        # Get the title and page number of the next section (if it exists)
        if i + 1 < len(toc_entries):
            end_title = toc_entries[i + 1]["title"]
            end_page = toc_entries[i + 1]["page_number"] - 1  # Adjust for zero-indexed
        else:
            # If it is the last section, set it to the last page
            end_page = len(page_contents) - 1

        # Ensure start_page and end_page are within valid range
        start_page = max(0, min(start_page, len(page_contents) - 1))
        end_page = max(0, min(end_page, len(page_contents) - 1))

        section_text = ""
        page_numbers = []

        # Iterate over the page range
        for page_num in range(start_page, end_page + 1):
            page_content = page_contents[page_num]

            if page_num == start_page:
                # Locate the starting point of the section
                start_positions = find_fuzzy_match_positions(page_content, start_title)
                if start_positions:
                    start_pos = start_positions[0][0]
                    section_text += page_content[start_pos:].strip() + "\n"
                else:
                    section_text += page_content.strip() + "\n"

            if page_num == end_page:  # It may be both the start and end page
                # Locate the ending point of the section
                end_positions = []
                if i < len(toc_entries) - 1:  # In case of the last TOC entry, there is no end_title
                    end_positions = find_fuzzy_match_positions(page_content, end_title)
                if end_positions:
                    end_pos = end_positions[0][0]
                    section_text += page_content[:end_pos].strip() + "\n"
                else:
                    section_text += page_content.strip() + "\n"
            
            if (page_num != start_page) and (page_num != end_page):
                # Middle pages, directly add the entire content
                section_text += page_content.strip() + "\n"

            page_numbers.append(page_num + 1)  # Store 1-indexed page numbers

        # Add the section content to the returned sections list
        sections.append({
            'uuid': get_uuid(name=f"{pdf_name}_title_{start_title}"),
            'title': start_title,
            'text': section_text.strip(),
            'page_numbers': page_numbers
        })

    return sections


def get_ai_research_per_page_table_info(metadata: str, pdf_data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """ Output (table_data):
        [ [ {'uuid': uuid1 of page1, 'table_content': table_html ,'table_caption': table caption1, 'bbox': bbox1, 'table_summary':table_summary}, {...} ], [ {...} ] ... ]
    """
    pdf_data_mineru = pdf_data["info_from_mineru"]
    pdf_path = metadata["pdf_path"]
    pdf_name = metadata["uuid"]
    results = []

    # Group tables by page number
    tables_by_page = {}
    for table in pdf_data_mineru.get("tables", []):
        page_number = table.get("page_number", 0)
        if page_number not in tables_by_page:
            tables_by_page[page_number] = []
        tables_by_page[page_number].append(table)

    # Iterate through each page of the PDF
    for page_num, page in enumerate(extract_pages(pdf_path), start=1):
        if page_num in tables_by_page:  # If the page has tables
            page_tables = tables_by_page[page_num]
            page_result = []

            for ordinal, table in enumerate(page_tables, start=1):
                # Generate UUID for the table
                uuid = get_uuid(name=f"{pdf_name}_page_{page_num}_table_{ordinal}")

                # Generate a summary for the table using LLM
                messages = [
                    {"role": "system", "content": "You are an expert in summarizing data. Your task is to generate a concise summary for an HTML-formatted table, focusing on key information and describing the table content clearly and succinctly."},
                    {"role": "user", "content": f"Please generate a brief summary for the following table without any extra information or formatting in no more than 50 words. \nTable Caption: {table['table_caption']}\nTable Content in html: {table['table_html']}"}
                ]
                table_summary = call_llm_with_message(messages)

                # Build the table info dictionary
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
        pdf_data: Dict[str, Any]
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
            image_summary = get_image_summary(image["figure_path"])
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
            equation_text = equation.get("eq_text", "")
            if equation_text != "":
                uuid = get_uuid(name=f"{pdf_name}_page_{page_num}_equation_{ordinal}")
                result.append({'uuid': uuid, 'text': equation_text})
        results.append(result)
    
    return results

# TODO: modify the following metadata function
def aggregate_ai_research_metadata(metadata: Dict[str, Any]) -> List[List[Any]]:
    """ Output:
        [ [ paper_id, paper_pages, paper_path ] ]
    """
    paper_id = metadata["uuid"]
    num_pages = metadata["num_pages"]
    pdf_path = metadata["pdf_path"]
    return [[paper_id, num_pages, pdf_path]]

def aggregate_ai_research_pages(metadata: Dict[str, Any], page_data: Dict[str, Any]) -> List[List[Any]]:
    """ Output:
        [ [ page_id, page_number, page_width, page_height, page_content, ref_paper_id ] ]
    """
    
    pdf_path = metadata["pdf_path"]
    pdf_name = metadata["uuid"]
    doc = fitz.open(pdf_path)
    results = []

    for page_number, page_content in enumerate(page_data["page_contents"], start=1):
        # Get the corresponding page from the PDF
        page = doc[page_number - 1]
        page_uuid = get_uuid(name=f"{pdf_name}_page_{page_number}")
        page_width = page.rect.width
        page_height = page.rect.height

        results.append([
            page_uuid,        # page_id
            page_number,      # page_number (starts from 1)
            int(page_width),  # page_width
            int(page_height), # page_height
            page_content,     # page_content
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
        [ [ section_id, section_title, section_text, section_ordinal, page_numbers, ref_paper_id ] ]
    """
    
    paper_id = metadata["uuid"]
    results = []

    for ordinal, section in enumerate(section_data):
        section_id = section["uuid"]
        section_title = section["title"]
        section_text = section["text"]
        page_numbers = section["page_numbers"]

        results.append([
            section_id,     # section_id
            section_title,  # section_title
            section_text,   # section_content
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
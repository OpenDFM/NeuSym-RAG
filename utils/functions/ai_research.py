#coding=utf8
import json, sys, os, re, logging
from typing import List, Union, Optional, Tuple, Any, Dict
import tempfile
import fitz  # PyMuPDF
import PyPDF2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage, LTRect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.functions.common_functions import get_uuid, call_llm_with_message
from utils.functions.pdf_functions import crop_pdf, convert_pdf_to_image,get_pdf_page_text
from utils.functions.image_functions import get_image_summary


def load_json_from_processed_data(pdf_path: str) -> dict:
    """output:
    {  
        "pdf_path": "data/dataset/airqa/papers/example.pdf",
        "info_from_mineru": {
            
            "tables": [
                {
                    "table_caption": "Table 1: Caption text here",
                    "table_html": "<table>...</table>",
                    "table_bbox": [50, 50, 150, 250],
                    "page_number": 1
                },
                {
                    "table_caption": "Table 2: Another caption",
                    "table_html": "<table>...</table>",
                    "table_bbox": [100, 100, 300, 400],
                    "page_number": 2
                }
            ]
        }
    }
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path=os.path.join(os.path.dirname(os.path.dirname(pdf_path)),'processed_data',f'{pdf_name}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    return pdf_data

def get_financial_report_per_page_page_uuid_and_info(pdf_path: str) -> Dict[str,Any]:
    """ Output:
        Dict[str, Union[str, List[str]]], the output dictionary containing the following keys:
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
            - page_contents: List[str], the list of strings, each string represents the content of each page.
            - page_uuids: List[str], the list of UUIDs for each page if generate_uuid is True.
    """
    return get_pdf_page_text(pdf_path)



def get_financial_report_per_page_chunk_uuid_and_text(pdf_path: str, page_data: dict, chunk_size: int, chunk_overlap: int) -> List[List[Dict[str, str]]]:
    """ 
    Output:
        [
            [{'uuid': uuid1, 'text': text1 of page 1}, {'uuid': uuid2, 'text': text2 of page 1}, ...],
            [{'uuid': uuid1, 'text': text1 of page 2}, {'uuid': uuid2, 'text': text2 of page 2}, ...],
            ...
        ]
    """
    # Assuming text_splitter is a predefined function that splits text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Initialize an empty list to store the results
    results = []
    
    # Extract the PDF name from the path (assuming the PDF name is the last part of the path, before extension)
    pdf_name = pdf_path.split('/')[-1].split('.')[0]
    
    # Iterate through each page's content and UUIDs
    for page_idx, page_content in enumerate(page_data['page_contents'], start=1):
        # Split the page's content into chunks
        page_chunks = text_splitter.create_documents([page_content])
        
        # For each chunk, generate a UUID based on the PDF name, page number, and chunk number
        page_results = []
        for chunk_idx, chunk in enumerate(page_chunks, start=1):
            # Create a UUID using the format: pdfname_pageX_chunkY
            chunk_uuid = f"{pdf_name}_page{page_idx}_chunk{chunk_idx}"
            
            # Add the UUID and text to the result
            page_results.append({'uuid': chunk_uuid, 'text': chunk.page_content})
        
        # Append the results for this page
        results.append(page_results)
    
    return results




def get_ai_research_per_page_table_uuid_and_info(pdf_path: str, pdf_data: dict) -> List[List[Dict[str, Any]]]:
    """
    Extracts per-page table information with UUID and summary.

    input:
        pdf_data (dict): The processed JSON structure for a PDF, containing table data.

    output:
        [ [ {'uuid': uuid1 of page1, 'table_content':table_html ,'table_caption':table caption1, 'bbox': bbox1, 'table_summary':table_summary}, {...} ], [ {...} ] ... ]
    """
    pdf_data_mineru=pdf_data["info_from_mineru"]
    output = []  # Final output containing tables grouped by pages

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
                uuid = get_uuid(name=f"page_{page_num}_table_{ordinal}")

                # Generate a summary for the table using LLM
                messages = [
                    {"role": "system", "content": "You are an expert in summarizing data. Your task is to generate a concise summary for an HTML-formatted table, focusing on key information and describing the table content clearly and succinctly."},
                    {"role": "user", "content": f"Please generate a brief summary for the following table without any extra information or formatting in no more than 50 words. \nTable Caption:{table[table_caption]}\nTable Content in html:{table[table_html]}"}
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

            output.append(page_result)  # Append tables for this page
        else:
            output.append([])  # Append empty list if no tables on this page

    return output




def get_ai_research_per_page_figure_uuid_and_summary(pdf_path: str) -> List[List[Dict[str, Any]]]:
    """ Output:
        [ [ {'uuid': uuid1, 'summary': summary1, 'bbox': bbox1}, {...} ], [ {...} ] ... ]
    """
    pdf_file_obj = open(pdf_path, 'rb')
    pdf_readed = PyPDF2.PdfReader(pdf_file_obj)
    tmp_pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', dir=os.path.join(os.getcwd(), '.cache'))
    tmp_png_file = tempfile.NamedTemporaryFile(suffix='.png', dir=os.path.join(os.getcwd(), '.cache'))
    results = []
    
    for page_num, page in enumerate(extract_pages(pdf_path), start = 1):
        page_obj = pdf_readed.pages[page_num - 1]
        page_width = page_obj.mediabox.upper_right[0]
        page_height = page_obj.mediabox.upper_right[1]
        page_data = []
        for element_num, element in enumerate(page._objs, start = 1):
            if isinstance(element, LTFigure):
                element.x1 = element.y1 = 0
                for sub_element in element:
                    if not isinstance(sub_element, LTRect) and sub_element.x1 <= page_width and sub_element.y1 <= page_height:
                        element.x1 = max(element.x1, sub_element.x1)
                        element.y1 = max(element.y1, sub_element.y1)
                element.x1 = min(element.x1 + 5, page_width)
                element.y1 = min(element.y1 + 5, page_height)
                element.y0, element.y1 = element.y1, element.y0
            if isinstance(element, (LTImage, LTFigure)):
                crop_pdf(element, page_obj, tmp_pdf_file.name)
                convert_pdf_to_image(tmp_pdf_file.name, tmp_png_file.name)
                uuid = get_uuid(f"page_{page_num}_figure_{element_num}")
                summary = get_image_summary(tmp_png_file.name)
                bbox = [element.x0, page_height - element.y1, element.x1 - element.x0, element.y1 - element.y0]
                page_data.append({'uuid': uuid, 'summary': summary, 'bbox': bbox})
        results.append(page_data)

    pdf_file_obj.close()
    tmp_pdf_file.close()
    tmp_png_file.close()
    
    return results
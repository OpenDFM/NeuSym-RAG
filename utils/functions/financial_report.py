#coding=utf8
import json, sys, os, re, logging
from typing import List, Dict, Union, Optional, Any, Iterable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.functions.common_functions import get_uuid,call_llm
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf #table extraction
import tempfile
from pdf2image import convert_from_path


def get_financial_report_per_page_uuid(pdf_data: dict) -> List[str]:
    """ Output:
        [ page_uuid1, page_uuid2, ...  ]
    """
    page_nums = [page_info['page_number'] for page_info in pdf_data['page_infos']]
    return [get_uuid(f"{pdf_data['pdf_id']}_page_{page_num}") for page_num in page_nums]


def get_financial_report_per_page_content_uuid(pdf_data: dict) -> List[List[str]]:
    """ Output:
        [ [content_uuid1, content_uuid2, ...], [content_uuid1, content_uuid2, ...], ... ]
    """
    page_nums = [page_info['page_number'] for page_info in pdf_data['page_infos']]
    return [[get_uuid(f"{pdf_data['pdf_id']}_page_{page_num}_content_{cid}") for cid in range(len(pdf_data['page_infos'][idx]['bbox_text']))] for (idx, page_num) in enumerate(page_nums)]


def get_financial_report_per_page_chunk_uuid_and_text(pdf_data: dict, chunk_size: int, chunk_overlap: int) -> List[List[Dict[str, str]]]:
    """ Output:
        [ [{'uuid': uuid1, 'text': text1}, {'uuid': uuid2, 'text': text2}, ...], [{'uuid': uuid1, 'text': text1}, {'uuid': uuid2, 'text': text2}, ...], ... ]
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    results = []
    for page_info in pdf_data['page_infos']:
        page_text = re.sub(r'\n+', '\n', '\n'.join(text.strip() for text in page_info['bbox_text']))
        page_chunks = text_splitter.create_documents([page_text])
        results.append([{'uuid': get_uuid(f"{pdf_data['pdf_id']}_page_{page_info['page_number']}_chunk_{idx}"), 'text': page_chunks[idx].page_content} for idx in range(len(page_chunks))])
    return results


def aggregate_financial_report_table_metadata(pdf_data: dict) -> List[List[Any]]:
    """ Output:
        [ [ report_id, report_pages, report_path ] ]
    """
    return [[pdf_data['pdf_id'], pdf_data['num_pages'], pdf_data['pdf_path']]]


def aggregate_financial_report_table_pages(pdf_data: dict, page_ids: List[str]) -> List[List[Any]]:
    """ Output:
        [ [ page_id, page_number, page_width, page_height, page_content, ref_report_id ] ]
    """
    results = []
    ref_report_id = pdf_data['pdf_id']
    for page_info, page_id in zip(pdf_data['page_infos'], page_ids):
        page_number = int(page_info['page_number'])
        results.append([page_id, page_number, page_info['width'], page_info['height'], re.sub(r'\s+', ' ', " ".join(page_info['bbox_text'])), ref_report_id])
    return results


def aggregate_financial_report_table_content(pdf_data: dict, page_ids: List[str], content_ids: List[List[str]]) -> List[Any]:
    """ Output:
        [ [ content_id, text_content, bounding_box, ordinal, ref_report_id, ref_page_id ] ]
    """
    results = []
    ref_report_id = pdf_data['pdf_id']
    for idx, (page_info, page_id) in enumerate(zip(pdf_data['page_infos'], page_ids)):
        for ordinal, (content_id, bbox, text) in enumerate(zip(content_ids[idx], page_info['bbox'], page_info['bbox_text'])):
            results.append([content_id, re.sub(r'\s+', ' ', text), json.dumps(bbox), ordinal, ref_report_id, page_id])
    return results


def aggregate_financial_report_table_chunks(pdf_data: dict, page_ids: List[str], chunks: List[List[Dict[str, str]]]) -> List[Any]:
    """ Output:
        [ [ chunk_id, text_content, ordinal, ref_report_id, ref_page_id ] ]
    """
    results = []
    ref_report_id = pdf_data['pdf_id']
    for idx, page_id in enumerate(page_ids):
        for ordinal, chunk in enumerate(chunks[idx]):
            results.append([chunk['uuid'], chunk['text'], ordinal, ref_report_id, page_id])
    return results


def get_financial_report_per_page_tableinpages(pdf_data: dict, model='gpt-4o', top_p=0.95, temperature=0.7) -> List[dict]:
    """
    Process pdf_data to extract tables and their bounding boxes (bbox) using 'unstructured' library, and also give a summary of each table using llm
    Output:
        [ { "page_number": int, "table_html": str, "table_bbox": list, "table_summary":str } ]
    """
    table_data = []
    pdf_path=pdf_data['pdf_path']
    
    def normalize_bbox(bbox, width_ratio, height_ratio):
        # original: [x0, y0, x1, y1]
        # after: [x0, y0, width, height]
        x0, y0, x1, y1 = bbox
        return [
            round(x0 * width_ratio, 1),
            round(y0 * height_ratio, 1),
            round((x1 - x0) * width_ratio, 1),
            round((y1 - y0) * height_ratio, 1)
        ]

    images = convert_from_path(pdf_path) #pdf2image (provide page size information for library unstructured)

    for page_info in pdf_data['page_infos']:
        page_number = page_info['page_number']
        width, height = page_info['width'], page_info['height']
        
        # extract tables in this page
        new_pdf_document = fitz.open()
        with fitz.open(pdf_path) as pdf_document:
            new_pdf_document.insert_pdf(pdf_document, from_page=page_number - 1, to_page= page_number - 1)# fitz starting from 0
        
        with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as temp_pdf:
            # write the new pdf to template file
            new_pdf_document.save(temp_pdf.name)
            new_pdf_document.close()

            # extract tables using library unstructured
            elements = partition_pdf(filename=temp_pdf.name, infer_table_structure=True, strategy='hi_res')
            tables_thispage = [el for el in elements if el.category == "Table"]

        if not tables_thispage:
            continue #skip page without tables

        # Process each table on the page
        for table in tables_thispage:

            # Extract table HTML 
            table_html = table.metadata.text_as_html
            # Extract table bounding boxes and adjust size
            width_ratio = width / images[page_number - 1].size[0]
            height_ratio = height / images[page_number - 1].size[1] # PyMuPDF vs unstructured(starting from 0)
            (x0, y0), _, (x1, y1), _ = table.metadata.coordinates.points
            table_bbox = normalize_bbox((x0,y0,x1,y1), width_ratio, height_ratio) 
            # Get table summary using llm
            template = f"""You are an expert in summarizing data. Your task is to generate a concise summary for an HTML-formatted table, focusing on key information and describing the table content clearly and succinctly.
            
            Please generate a brief summary for the following HTML table content:
            {table_html}
            """
            summary = call_llm(template=template, model=model, top_p=top_p, temperature=temperature)

            # Add the result to the output
            table_data.append({
                "page_number": page_number,
                "table_html": table_html,
                "table_bbox": table_bbox,
                "table_summary": summary
            })

    return table_data


def get_financial_report_per_page_tableinpages_uuid(pdf_data: dict, table_data: List[dict]) -> List[List[str]]:
    """
    Generate unique UUIDs for tables in each page.
    Output:
        [ [table_uuid1_page1, table_uuid2_page1, ...], [table_uuid1_page2, ...], ... ]
    """
    results = []
    
    for page_info in pdf_data['page_infos']:
        page_number = page_info['page_number']
        tables_in_page = [table for table in table_data if table['page_number'] == page_number]
        
        # Generate UUIDs only for tables in the current page
        table_uuids = [
            get_uuid(f"{pdf_data['pdf_id']}_page_{page_number}_table_{idx}") 
            for idx, table in enumerate(tables_in_page)
        ]
        
        results.append(table_uuids)
    
    return results




def aggregate_financial_report_table_tableinpages(pdf_data: dict, table_data: List[dict], page_ids: List[str], table_ids: List[List[str]]) -> List[Any]:
    """ 
    Aggregate table data from multiple pages.
    Output:
        [ [ table_id, table_html, table_bbox, table_ordinal, ref_paper_id, ref_page_id ] ]
    """
    results = []
    ref_paper_id = pdf_data['pdf_id']  # The reference paper ID from the PDF data

    for page_idx, (page_info, page_id) in enumerate(zip(pdf_data['page_infos'], page_ids)):
        page_number = page_info['page_number']
        tables_in_page = [table for table in table_data if table['page_number'] == page_number]

        if not tables_in_page:
            continue  # Skip if there are no tables in the current page

        for table_idx, table in enumerate(tables_in_page):
            table_id = table_ids[page_idx][table_idx]  # Use pre-generated table UUID from get_financial_report_per_page_tableinpages_uuid
            table_html = table["table_html"]
            table_bbox = table["table_bbox"]
            ordinal = table_idx  # The order of the table on the page
            table_summary = table["table_summary"]

            # Append the result for the current table
            results.append([table_id, table_html, table_summary, json.dumps(table_bbox), ordinal, ref_paper_id, page_id])

    return results



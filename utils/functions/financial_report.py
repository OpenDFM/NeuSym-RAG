#coding=utf8
import json, sys, os, re, logging
from typing import List, Dict, Union, Optional, Any, Iterable
from utils.functions.common_functions import get_uuid


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


def aggregate_financial_report_table_metadata(pdf_data: dict) -> List[Any]:
    """ Output:
        [ [ paper_id, num_pages, pdf_path ] ]
    """
    return [[pdf_data['pdf_id'], pdf_data['num_pages'], pdf_data['pdf_path']]]


def aggregate_financial_report_table_pages(pdf_data: dict, page_ids: List[str]) -> List[Any]:
    """ Output:
        [ [ page_id, page_number, page_width, page_height, page_content, ref_paper_id ] ]
    """
    results = []
    ref_paper_id = pdf_data['pdf_id']
    for page_info, page_id in zip(pdf_data['page_infos'], page_ids):
        page_number = int(page_info['page_number'])
        results.append([page_id, page_number, page_info['width'], page_info['height'], re.sub(r'\s+', ' ', " ".join(page_info['bbox_text'])), ref_paper_id])
    return results


def aggregate_financial_report_table_content(pdf_data: dict, page_ids: List[str], content_ids: List[List[str]]) -> List[Any]:
    """ Output:
        [ [ content_id, text_content, bounding_box, ordinal, ref_paper_id, ref_page_id ] ]
    """
    results = []
    ref_paper_id = pdf_data['pdf_id']
    for idx, (page_info, page_id) in enumerate(zip(pdf_data['page_infos'], page_ids)):
        for ordinal, (content_id, bbox, text) in enumerate(zip(content_ids[idx], page_info['bbox'], page_info['bbox_text'])):
            results.append([content_id, re.sub(r'\s+', ' ', text), json.dumps(bbox), ordinal, ref_paper_id, page_id])
    return results
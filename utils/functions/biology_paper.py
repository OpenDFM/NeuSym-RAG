#coding=utf8
import json, sys, os, re, logging
from typing import List, Dict, Union, Optional, Any, Iterable
from utils.functions.common_functions import get_uuid


def get_per_page_uuid(pdf_data: dict) -> List[str]:
    """ Output:
        [ page_uuid1, page_uuid2, ...  ]
    """
    num_pages = pdf_data['num_pages']
    return [get_uuid(f"{pdf_data['pdf_id']}_page_{page_num}") for page_num in range(1, num_pages + 1)]


def get_per_page_content_uuid(pdf_data: dict) -> List[List[str]]:
    """ Output:
        [ [content_uuid1, content_uuid2, ...], [content_uuid1, content_uuid2, ...], ... ]
    """
    num_pages = pdf_data['num_pages']
    return [[get_uuid(f"{pdf_data['pdf_id']}_page_{page_id}_content_{cid}") for cid in range(len(pdf_data['page_infos'][page_id - 1]['bbox_text']))] for page_id in range(1, num_pages + 1)]


def aggregate_biology_paper_table_metadata(pdf_data: dict) -> List[Any]:
    """ Output:
        [ [ paper_id, num_pages, local_folder ] ]
    """
    return [[pdf_data['pdf_id'], pdf_data['num_pages'], os.path.join('data', 'dataset', 'pdfvqa', 'processed_data', 'bbox_images')]]


def aggregate_biology_paper_table_pages(pdf_data: dict, uuids: List[int]) -> List[Any]:
    """ Output:
        [ [ page_id, figure_path, page_number, page_width, page_height, page_content, ref_paper_id ] ]
    """
    results = []
    ref_paper_id = pdf_data['pdf_id']
    for page_info, uid in zip(pdf_data['page_infos'], uuids):
        page_number = int(page_info['page_number'])
        figure_path = os.path.join('data', 'dataset', 'pdfvqa', 'processed_data', 'bbox_images', f'{ref_paper_id}_{page_number}.png')
        if os.path.exists(figure_path):
            results.append([uid, figure_path, page_number, page_info['width'], page_info['height'], "\n".join(page_info['bbox_text']), ref_paper_id])
    return results


def aggregate_biology_paper_table_content_types() -> List[Any]:
    """ Output:
        [ [ type_id, type_name ] ]
    """
    return [[1, 'main text'], [2, 'section title'], [3, 'reference list'], [4, 'tables in the paper'], [5, 'figures in the paper']]


def aggregate_biology_paper_table_content(pdf_data: dict, page_ids: List[str], content_ids: List[List[str]]) -> List[Any]:
    """ Output:
        [ [ content_id, content_type, text_content, bounding_box, ordinal, ref_paper_id, ref_page_id ] ]
    """
    results = []
    ref_paper_id = pdf_data['pdf_id']
    for idx, (page_info, page_id) in enumerate(zip(pdf_data['page_infos'], page_ids)):
        for ordinal, (content_id, content_type, bbox, text) in enumerate(zip(content_ids[idx], page_info['bbox_label'], page_info['bbox'], page_info['bbox_text'])):
            results.append([content_id, content_type, text, json.dumps(bbox), ordinal, ref_paper_id, page_id])
    return results


def aggregate_biology_paper_table_parent_child_relations(pdf_data: dict, content_ids: List[List[str]]) -> List[Any]:
    """ Output:
        [ [ parent_id, child_id ] ]
    """
    results = []
    for idx, page_info in enumerate(pdf_data['page_infos']):
        content_id_mappings = content_ids[idx]
        for parent_id, child_id in page_info['relations']:
            results.append([content_id_mappings[parent_id], content_id_mappings[child_id]])
    return results
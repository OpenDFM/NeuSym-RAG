#coding=utf8
import json, sys, os, re, logging
from typing import List, Any, Dict
import tempfile

import PyPDF2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage, LTRect

from utils.functions.common_functions import get_uuid
from utils.functions.pdf_functions import crop_pdf, convert_pdf_to_image
from utils.functions.image_functions import get_image_summary


def get_ai_research_per_page_uuid(pdf_data: dict) -> List[str]:
    pass

def get_ai_research_per_page_figure_uuid_and_summary(pdf_data: dict) -> List[List[Dict[str, Any]]]:
    """ Output:
        [ [ {'uuid': uuid1, 'summary': summary1, 'bbox': bbox1}, {...} ], [ {...} ] ... ]
    """
    pdf_path, pdf_uuid = pdf_data['pdf_path'], pdf_data['uuid']
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
                uuid = get_uuid(f"{pdf_uuid}_page_{page_num}_figure_{element_num}")
                summary = get_image_summary(tmp_png_file.name)
                bbox = [element.x0, page_height - element.y1, element.x1 - element.x0, element.y1 - element.y0]
                page_data.append({'uuid': uuid, 'summary': summary, 'bbox': bbox})
        results.append(page_data)

    pdf_file_obj.close()
    tmp_pdf_file.close()
    tmp_png_file.close()
    
    return results

def aggregate_ai_research_table_figures(
        pdf_data: dict, 
        page_ids: List[str], 
        figures: List[List[Dict[str, Any]]]
    ) -> List[List[Any]] :
    """ Output:
        [ [ figure_id, figure_summary, bounding_box, ordinal, ref_paper_id, ref_page_id ] ]
    """
    results = []
    ref_paper_id = pdf_data['uuid']
    
    for idx, page_id in enumerate(page_ids):
        for ordinal, figure in enumerate(figures[idx]):
            results.append([figure['uuid'], figure['summary'], json.dumps(figure['bbox']), ordinal, ref_paper_id, page_id])
    
    return results
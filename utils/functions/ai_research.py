#coding=utf8
import json, sys, os, re, logging
from typing import List, Any

import PyPDF2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage, LTRect

from utils.functions.common_functions import get_uuid
from utils.functions.pdf_functions import crop_pdf, convert_pdf_to_image

def get_ai_research_per_page_figure_uuid_and_summary(pdf_path: str) -> List[str]:
    """ Output:
        [ [ {'uuid': uuid1, 'summary': summary1, 'bbox': bbox1}, {...} ], [ {...} ] ... ]
    """
    pdf_file_obj = open(pdf_path, 'rb')
    pdf_readed = PyPDF2.PdfReader(pdf_file_obj)
    
    for page_num, page in enumerate(extract_pages(pdf_path), start = 1):
        page_obj = pdf_readed.pages[page_num - 1]
        page_width = page_obj.mediabox.upper_right[0]
        page_height = page_obj.mediabox.upper_right[1]
        for element_num, element in enumerate(page._objs, start = 1):
            output_pdf_dir = f'./tmp/page_{page_num}_figure_{element_num}.pdf'
            output_png_dir = f'./tmp/page_{page_num}_figure_{element_num}.png'
            if isinstance(element, LTImage):
                crop_pdf(element, page_obj, output_pdf_dir)
                convert_pdf_to_image(output_pdf_dir, output_png_dir)
            elif isinstance(element, LTFigure):
                element.x1 = element.y1 = 0
                for sub_element in element:
                    if not isinstance(sub_element, LTRect) and sub_element.x1 <= page_width and sub_element.y1 <= page_height:
                        element.x1 = max(element.x1, sub_element.x1)
                        element.y1 = max(element.y1, sub_element.y1)
                element.x1 = min(element.x1 + 5, page_width)
                element.y1 = min(element.y1 + 5, page_height)
                element.y0, element.y1 = element.y1, element.y0
                crop_pdf(element, page_obj, output_pdf_dir)
                convert_pdf_to_image(output_pdf_dir, output_png_dir)

    pdf_file_obj.close()
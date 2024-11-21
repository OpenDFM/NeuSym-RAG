#coding=utf8

import os, openai, uuid, re, sys, logging, subprocess, tempfile, json
from typing import List, Dict, Union, Optional, Any, Iterable

import fitz  # PyMuPDF
import PyPDF2
from PyPDF2 import PdfWriter, PageObject
from pdf2image import convert_from_path
from pdfminer.layout import LTImage, LTFigure, LTRect
from pix2text import Pix2Text
from pix2text.layout_parser import ElementType

from utils.functions.common_functions import call_llm, get_uuid

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
        top_p: float = 0.95
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

def crop_pdf(
        element: Union[LTFigure, LTImage],
        page_obj: PageObject,
        output_file: str
    ):
    """Crop a PDF file according to the bounding box of the element and save it to a new PDF file.

    @args:
        element: Union[LTFigure, LTImage], the element to be cropped.
        page_obj: PageObject, PDF-page object resolved from PyPDF2.
        output_file: str, path to output PDF file. 
    """
    [image_left, image_top, image_right, image_bottom] = [element.x0, element.y0, element.x1, element.y1]
    page_obj.mediabox.upper_left = (image_left, image_top)
    page_obj.mediabox.lower_right = (image_right, image_bottom)
    cropped_pdf_writer = PdfWriter()
    cropped_pdf_writer.add_page(page_obj)
    with open(output_file, 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)
    cropped_pdf_writer.close()

def convert_pdf_to_image(
        input_file: str,
        output_file: str,
        dpi: int = 1200
    ):
    """Convert a single-page PDF file to a PNG file.
    @args:
        input_file: str, the path to input PDF.
        output_file: str, the path to output PNG.
        dpi: int, image quality in DPI, default to 1200.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"PDF file {input_file} not found.")
    images = convert_from_path(input_file, dpi=dpi)
    image = images[0]
    image.save(output_file, "PNG")

ocr = None

def get_pdf_formula(
        pdf_path: str
    ) -> List[str] :
    """Get the list of all the formula in the pdf in latex format using pix2text.

    @args:
        pdf_path: str, the path to the pdf file.

    @returns:
        result: List[str], the list of all formula
    """
    
    if ocr == None:
        ocr = Pix2Text()
    
    result = []
    tmp_png_file = tempfile.NamedTemporaryFile(suffix='.png', dir=os.path.join(os.getcwd(), '.cache'))
    images = convert_from_path(pdf_path)
    for image in images:
        image.save(tmp_png_file.name, "PNG")
        elements = ocr(tmp_png_file.name).elements
        for element in elements:
            if element.type == ElementType.FORMULA:
                result.append(str(element.text))
    tmp_png_file.close()
    
    return result

def parse_pdf(
        pdf_path: str,
        processed_data_folder: str = 'data/dataset/airqa/processed_data'
    ) -> bool:
    """Parse a PDF file with MinerU.

    @return:
    `True` if the PDF has been fully parsed before. `False` otherwise.
    
    @output:
    The function generates a JSON file for a PDF file, containing the extracted table and figure information. 
    The output JSON structure is as follows:

    {
        "pdf_path": <str>,  # Path to the input PDF file
        "info_from_mineru": {
            "tables": [  # A list of extracted table information
                {
                    "table_caption": <str>,  # The caption of the table (if available)
                    "table_html": <str>,     # The table content in HTML format (if available)
                    "table_bbox": [<float>, <float>, <float>, <float>],  # The bounding box of the table in [x1, y1, x2, y2]
                    "page_number": <int>     # The page number where the table is located
                },
                ...
            ],
            "figures": [  # A list of extracted figure information
                {
                    "figure_caption": <str>,  # The caption of the figure (if available)
                    "figure_path": <str>,     # Path to the extracted image file
                    "figure_bbox": [<float>, <float>, <float>, <float>],  # The bounding box of the figure in [x1, y1, x2, y2]
                    "page_number": <int>      # The page number where the figure is located
                },
                ...
            ],
            "equations": [  # A list of extracted equation information
                {
                    "eq_text": <str>,  # The content of the equation in latex
                    "page_number": <int>      # The page number where the equation is located
                },
                ...
            ]
        },
        "TOC": [  # Table of Contents
            {
                "level": <int>,  # The level of the TOC entry
                "title": <str>,  # The title of the TOC entry
                "page_number": <int>  # The page number of the TOC entry
            },
            ...
        ]
    }

    Each JSON file is saved with the same name as the corresponding PDF in the `processed_data_folder`.
    For example, if the PDF is named `64148d31-f547-5f8c-a7a0-3cc080a195dd.pdf`, the output will be saved as `processed_data_folder/64148d31-f547-5f8c-a7a0-3cc080a195dd.json`.

    """
    
    # 1. Construct the command
    
    # First, install magic-pdf package by `pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com`
    # Second, download models according to https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_en.md
    # Third, modify `magic-pdf.json` in 
    #     "C:\Users\username" (windows) 
    #  or "/home/username"    (linux) 
    #  or "/Users/username"   (macos) 
    # to set the table-config to true
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(processed_data_folder, f'{pdf_name}.json')
    if os.path.exists(output_path):
        logger.info(f"PDF {pdf_name} has been fully processed before.")
        return True
        
    json_mid_path = os.path.join(processed_data_folder, f'{pdf_name}', 'auto', f'{pdf_name}_middle.json')
    json_con_path = os.path.join(processed_data_folder, f'{pdf_name}', 'auto', f'{pdf_name}_content_list.json')
    if os.path.exists(json_mid_path) and os.path.exists(json_con_path):
        logger.info(f"PDF {pdf_name} has been partially processed before.")
    else:
        logger.info(f"Processing PDF {pdf_name} with MinerU.")
        command = [
            "magic-pdf",
            "-p", pdf_path,                # input pdf_file
            "-o", processed_data_folder,   # output folder
            "-m", "auto"                   # method (ocr, txt, or auto)
        ]
        
        try:
            result = subprocess.run(command, check=True, text=True, encoding='UTF-8')
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(f"Command execution failed for {pdf_path} with error: {e.stderr}")
    
    
    # 2. Generate the JSON file
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_mid_path = os.path.join(processed_data_folder, f'{pdf_name}', 'auto', f'{pdf_name}_middle.json')
    json_con_path = os.path.join(processed_data_folder, f'{pdf_name}', 'auto', f'{pdf_name}_content_list.json')

    # Check if the JSON file exists
    if not os.path.exists(json_mid_path):
        raise FileNotFoundError(f"File {json_mid_path} does not exist")
    if not os.path.exists(json_con_path):
        raise FileNotFoundError(f"File {json_con_path} does not exist")
    with open(json_mid_path, 'r', encoding='utf-8') as f:
        mid_data = json.load(f)
    with open(json_con_path, 'r', encoding='utf-8') as g:
        content_data = json.load(g)

    # Initialize the output data structure
    result = {
        "pdf_path": pdf_path,
        "info_from_mineru": {
            "tables": [],
            "figures": [],
            "equations": [],
            "references": []
        },
        "TOC": []  
    }

    # Extract Table of Contents using PyMuPDF
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    for entry in toc:
        level, title, page = entry
        result["TOC"].append({
            "level": level,
            "title": title,
            "page_number": page
        })
    doc.close()

    # Iterate through the parsed results for each page of the PDF
    for page_number, page in enumerate(mid_data.get("pdf_info", []), start=1):
        # Extract information about tables
        for block in page.get("tables", []):
            if block.get("type") == "table":
                table_info = {
                    "table_caption": "",
                    "table_html": "",
                    "table_bbox": block.get("bbox", []),
                    "page_number": page_number
                }

                for sub_block in block.get("blocks", []):
                    if sub_block.get("type") == "table_caption":
                        table_info["table_caption"] += " ".join(
                            span.get("content", "")
                            for line in sub_block.get("lines", [])
                            for span in line.get("spans", [])
                        )
                    elif sub_block.get("type") == "table_body":
                        table_info["table_html"] = sub_block.get("lines", [{}])[0].get("spans", [{}])[0].get("html", "")

                result["info_from_mineru"]["tables"].append(table_info)

        # Extract information about figures
        for block in page.get("images", []):
            if block.get("type") == "image":
                figure_info = {
                    "figure_caption": "",
                    "figure_path": "",
                    "figure_bbox": block.get("bbox", []),
                    "page_number": page_number
                }

                for sub_block in block.get("blocks", []):
                    if sub_block.get("type") == "image_caption":
                        figure_info["figure_caption"] += " ".join(
                            span.get("content", "")
                            for line in sub_block.get("lines", [])
                            for span in line.get("spans", [])
                        )
                    elif sub_block.get("type") == "image_body":
                        image_path = sub_block.get("lines", [{}])[0].get("spans", [{}])[0].get("image_path", "")
                        figure_info["figure_path"] = os.path.join(processed_data_folder, f'{pdf_name}', 'auto', 'images', image_path)

                result["info_from_mineru"]["figures"].append(figure_info)

    # Extract information about equations
    
    references = []
    record_reference = 0
    
    for content in content_data:
        if content["type"]== "equation":
            eq_info={
                "eq_text": content["text"],
                "page_number": content["page_idx"] + 1
            }
            result["info_from_mineru"]["equations"].append(eq_info)
        if content["type"] == "text" and content.get("text_level", None) == 1:
            if content["text"].lower().startswith("reference"):
                record_reference = 1
                continue
            elif record_reference == 1:
                record_reference = 0
        if record_reference and content.get("text", None) != None:
            reference_list = list(str(content["text"]).split("\n"))
            if reference_list:
                references.extend(reference_list)

    result["info_from_mineru"]["references"] = [{"reference_text": reference} for reference in references if reference != ""]

    # Write each paper's data into a separate JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    return False
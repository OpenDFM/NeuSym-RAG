#coding=utf8

import os, openai, uuid, re, sys, logging, subprocess, tempfile, json
from typing import List, Dict, Union, Optional, Any, Iterable, Tuple
from difflib import SequenceMatcher
import fitz  # PyMuPDF
import PyPDF2
from PyPDF2 import PdfWriter, PageObject
from pdf2image import convert_from_path
from pdfminer.layout import LTImage, LTFigure, LTRect
from pix2text import Pix2Text
from pix2text.layout_parser import ElementType

from utils.functions.common_functions import call_llm, get_uuid, call_llm_with_message

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

def get_table_summary(
        table: Dict[str, Any],
        max_length: int = 50,
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
    prompt_template = """You are an expert in summarizing data. Your task is to generate a concise summary for an HTML-formatted table, focusing on key information and describing the table content clearly and succinctly.

Please generate a brief summary for the following table without any extra information or formatting in no more than {max_length} words. \nTable Caption: {table_caption}\nTable Content in html: {table_html}\nHere is your summary:
"""
    table_summary = call_llm(prompt_template.format(max_length=max_length, table_caption=table['table_caption'], table_html=table['table_html']), model=model, top_p=top_p, temperature=temperature)
    return table_summary

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

def get_pdf_formula(
        pdf_path: str
    ) -> List[str] :
    """Get the list of all the formula in the pdf in latex format using pix2text.

    @args:
        pdf_path: str, the path to the pdf file.

    @returns:
        result: List[str], the list of all formula
    """
    
    ocr = Pix2Text()
    
    result = []
    tmp_png_file = tempfile.mktemp(suffix='.png', dir=os.path.join(os.getcwd(), '.cache'))
    images = convert_from_path(pdf_path)
    for image in images:
        image.save(tmp_png_file, "PNG")
        elements = ocr(tmp_png_file).elements
        for element in elements:
            if element.type == ElementType.FORMULA:
                result.append(str(element.text))
    os.remove(tmp_png_file)
    
    return result

def parse_pdf(
        pdf_path: str,
        processed_data_folder: str = 'data/dataset/airqa/processed_data',
        TOC_threshold: float = 0.9
    ) -> bool:
    """Parse a PDF file with MinerU.
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
                    "equation_text": <str>,   # The content of the equation in latex
                    "page_number": <int>      # The page number where the equation is located
                },
                ...
            ],
            "references": [
                {
                    "reference": <str>  # The content of the reference
                },
                ...
            ],
            "TOC": [  # Table of Contents
                {
                    "level": <int>,  # The level of the TOC entry
                    "title": <str>,  # The title of the TOC entry
                    "text": <str>,  # The text of the relevant section
                    "page_number": <int>,  # The page number of the TOC entry
                    "page_numbers": [<int>,<int>]  # The page numbers of the secion
                },
                ...
            ]
        }
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
        return
        
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
            result = subprocess.run(command, check=True, text=True, encoding='utf-8')
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
            "references": [],
            "TOC": []
        }
    }

    # Extract Table of Contents using PyMuPDF
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()

    if toc:  # The case when PyMuPDF can parse TOC
        # Filter out level 1 entries directly while appending to result
        for entry in toc:
            level, title, page = entry
            if level == 1 and title:  # Only include level 1 entries
                result["info_from_mineru"]["TOC"].append({
                    "title": title,
                    "level": level,
                    "text": "",
                    "page_number": page,
                    "page_numbers": []
                })

        # Extract text of all pages
        page_contents = [doc[page_number].get_text() for page_number in range(doc.page_count)]

        # Fuzzy matching function
        def find_fuzzy_match_positions(content: str, target: str, threshold: float) -> List[Tuple[int, int]]:
            """
            Perform fuzzy matching to find positions of the target in the content.
            Returns a list of start and end positions for matches.
            """
            lines = content.splitlines()
            matches = []
            for line in lines:
                similarity = SequenceMatcher(None, line.lower(), target.lower()).ratio()
                if similarity > threshold:
                    match = re.search(re.escape(line), content)
                    if match:
                        matches.append((match.start(), match.end()))  # Store start and end positions
            return matches

        # Process all level 1 TOC entries
        toc_entries = result["info_from_mineru"]["TOC"]

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
                    start_positions = find_fuzzy_match_positions(page_content, start_title, threshold=TOC_threshold)
                    if start_positions:
                        start_pos = start_positions[0][0]
                        section_text += page_content[start_pos:].strip() + "\n"
                    else:
                        section_text += page_content.strip() + "\n"

                if page_num == end_page:
                    # Locate the ending point of the section
                    end_positions = []
                    if i < len(toc_entries) - 1:  # In case of the last TOC entry, there is no end_title
                        end_positions = find_fuzzy_match_positions(page_content, end_title, threshold=TOC_threshold)
                    if end_positions:
                        end_pos = end_positions[0][0]
                        section_text += page_content[:end_pos].strip() + "\n"
                    else:
                        section_text += page_content.strip() + "\n"

                if (page_num != start_page) and (page_num != end_page):
                    # Middle pages, directly add the entire content
                    section_text += page_content.strip() + "\n"

                page_numbers.append(page_num + 1)  # Store 1-indexed page numbers

            # Update TOC entry
            toc_entry["text"] = section_text.strip()
            toc_entry["page_numbers"] = sorted(page_numbers)  # Sort page numbers for consistency

    else:  # If pymupdf cannot parse cot, Use result parsed by mineru

        # Filter out TOC entries with text_level == 1
        toc_entries = [entry for entry in content_data if entry.get("type") == "text" and entry.get("text_level") == 1]

        for i, toc_entry in enumerate(toc_entries):
            # Create a new TOC entry
            title = toc_entry["text"].strip()
            start_idx = content_data.index(toc_entry)  # Current TOC entry index
            start_page = toc_entry["page_idx"] + 1  # Convert to 1-based page number

            # Determine the range of content for this TOC entry
            if i + 1 < len(toc_entries):
                end_idx = content_data.index(toc_entries[i + 1])  # Next TOC entry index
            else:
                end_idx = len(content_data)  # Until the end of content_data

            # Gather all content between start_idx and end_idx
            section_text = ""
            page_numbers = set()

            for entry in content_data[start_idx + 1 : end_idx]:  # Exclude the current TOC entry
                if entry.get("type") == "text":
                    section_text += entry["text"].strip() + "\n"
                    page_numbers.add(entry["page_idx"] + 1)  # Convert to 1-based page number

            # Add the TOC entry to the result
            section_text = section_text.strip()
            if title or section_text:
                result["info_from_mineru"]["TOC"].append({
                    "title": title,
                    "text": section_text,
                    "level": toc_entry["text_level"],
                    "page_number": start_page,
                    "page_numbers": sorted(page_numbers)  # Ensure page numbers are sorted
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
                    elif sub_block.get("type") == "table_footnote":
                        table_info["table_caption"] += " ".join(
                            span.get("content", "")
                            for line in sub_block.get("lines", [])
                            for span in line.get("spans", [])
                        )
                    elif sub_block.get("type") == "table_body":
                        table_info["table_html"] = sub_block.get("lines", [{}])[0].get("spans", [{}])[0].get("html", "")
                
                if table_info["table_html"]:
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

                if figure_info["figure_path"]:
                    result["info_from_mineru"]["figures"].append(figure_info)

    # Extract information about equations
    
    references = []
    record_reference = 0
    
    for content in content_data:
        if content["type"]== "equation":
            equation_info={
                "equation_text": content["text"],
                "page_number": content["page_idx"] + 1
            }
            if equation_info["equation_text"]:
                result["info_from_mineru"]["equations"].append(equation_info)
        if content["type"] == "text" and content.get("text_level", None) == 1:
            if content["text"].lower().startswith("reference"):
                record_reference = 1
                continue
            elif record_reference == 1:
                record_reference = 0
        if record_reference and content.get("text", None) != None:
            reference_list = list(str(content["text"]).split("\n"))
            reference_list = [reference.strip() for reference in reference_list if reference.strip()]
            if reference_list:
                references.extend(reference_list)

    result["info_from_mineru"]["references"] = [{"reference_text": reference} for reference in references if reference]

    # Write each paper's data into a separate JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    return result

def load_json_from_processed_data(
        pdf_path: str, 
        processed_data_folder: str,
        TOC_threshold: float
    ) -> dict:
    """ Load the parsed JSON data from a PDF file. See `parse_pdf` for more details.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(processed_data_folder, f'{pdf_name}.json')
    if not os.path.exists(json_path):
        pdf_data = parse_pdf(pdf_path=pdf_path, processed_data_folder=processed_data_folder, TOC_threshold=TOC_threshold)
    with open(json_path, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    return pdf_data
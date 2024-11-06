#coding=utf8
import os, json, sys, logging, re, tqdm, math, zipfile, pickle, random
import pandas as pd
import fitz  # PyMuPDF
import shutil
from fuzzywuzzy import fuzz  # For fuzzy matching
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Union, Optional, Tuple, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions.common_functions import get_uuid
from utils.functions.image_functions import draw_image_with_bbox
from utils.functions.pdf_functions import get_pdf_page_text


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dataset')


def process_pdfvqa(
        raw_data_folder: str = 'data/dataset/pdfvqa/raw_data',
        processed_data_folder: str = 'data/dataset/pdfvqa/processed_data',
        image_folder_name: str = 'bbox_images',
        test_data_name: str = 'test_data.jsonl',
        pdf_data_name: str = 'pdf_data.jsonl'
    ):
    """ Process the PDFVQA dataset into a unified format and remove duplication.
    @param:
        raw_data_folder: str, the path to the raw data folder.
        processed_data_dir: str, the path to the processed data directory.
        image_folder_name: str, the folder name for the image data (.png decorated with bounding boxes and numeric labels), default is 'bbox_images'.
        test_data_name: str, the name of the processed test data file, default is 'test_data.jsonl'.
        pdf_data_name: str, the name of the processed pdf data file, default is 'pdf_data.jsonl'.
    @return:
        test_data: List[Dict[str, Any]], return the list of processed data, each data point is a dictionary containing the following fields:
            {
                "uuid": "xxx-xxx-xxx-xxx", // str, UUID of the question
                "task_type": "a", // str, chosen from a, b, c
                "question": "Is it correct that there is no figure on the top left?", // str, the question text
                "question_type": "existence", // str, chosen from [existence, object_recognition, structural_understanding, parent_relationship_understanding, child_relationship_understanding]
                "answer": true, // Union[bool, str, List[str]], three types of answers for different task types
                "pdf_id": "15450119", // str, UUID of the PDF file
                "page_number": 3 // int, the reference page number for this question (used to extend the question context), starting from 1. Note that this field is None for task c
            }
        pdf_data: List[Dict[str, Any]], return the list of processed pdf files, each data point is a dictionary containing the following fields:
            {
                "pdf_id": "15450119", // str, UUID of the PDF file
                "num_pages": 9, // int, number of PDF pages
                "page_infos": [ // List[Dict[str, Any]], information of each page, which is a dictionary containing the following fields:
                    {
                        "page_number": 1, // int, the page number, starting from 1
                        "width": 640, // int, the width of the page
                        "height": 780, // int, the height of the page
                        "page_path": "data/dataset/pdfvqa/processed_data/bbox_images/15450119_1.png",
                        "bbox": [ // List[Tuple[float, float, float, float]], [x0, y0, width, height]
                            [0, 0, 340, 230],
                            ...
                        ], // OCR bounding boxes in the current page, 4-tuple List
                        "bbox_text": [
                            "Text content of the first bbox.",
                            ...
                        ], // List[str], the OCR text content of the current page
                        "bbox_label": [
                            1,
                            ...
                        ], // List[int], labels for each bbox, 1->main text, 2->section title, 3->list such as references, 4->tables, 5->figures
                        "relations": [ // List[Tuple[int, int]], records the parent-child relations between different bbox in the current page
                            [0, 1], // numbers represent bbox index, starting from 0
                            ...
                        ] // the first element is the parent, while the second is the child
                    },
                    ... // other pages
                ]
            }
    """
    image_zip = os.path.join(raw_data_folder, 'test_images.zip')
    if not os.path.exists(image_zip):
        raise FileNotFoundError(f"File {image_zip} not found.")
    with zipfile.ZipFile(image_zip, 'r') as zip_ref:
        zip_ref.extractall(processed_data_folder) # data/dataset/pdfvqa/processed_data/test_images/
    raw_image_folder = os.path.join(processed_data_folder, 'test_images')

    # output image folder with bounding boxes: data/dataset/pdfvqa/processed_data/bbox_images/
    image_folder = os.path.join(processed_data_folder, image_folder_name)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    pdf_data = []
    docs = pickle.load(open(os.path.join(raw_data_folder, 'test_doc_info_visual.pkl'), 'rb'))

    empty_page_dict = lambda x: {"page_path": "", "width": 0, "height": 0, "page_number": x, "bbox": [], "bbox_text": [], "bbox_label": [], "relations": []}
    errata = json.load(open(os.path.join(processed_data_folder, 'errata.json'), 'r', encoding='UTF-8'))
    # preprocess images of each PDF page
    for pdf_id in docs:
        tmp_pdf_data = {}
        tmp_pdf_data['pdf_id'] = pdf_id
        # special care to pdf with id "28649313", missing the first 16 pages
        tmp_pdf_data['num_pages'] = len(docs[pdf_id]['pages']) if pdf_id != '28649313' else 21
        tmp_pdf_data['page_infos'] = [] if pdf_id != '28649313' else [empty_page_dict(idx + 1) for idx in range(16)]
        pdf_data.append(tmp_pdf_data)

        pages = docs[pdf_id]['pages']
        for page_id in sorted(pages.keys()): # pre-process each page and bounding boxes in it
            image_name = pages[page_id]['name'] # e.g., {pdf_id}.pdf_{page_id}.png
            image_pdf_id, image_page_id = image_name.split('.')[0], image_name.split('.png')[0].split('_')[-1]
            assert str(pdf_id) == image_pdf_id and str(page_id) == image_page_id, f"PDF id |{pdf_id}| != Image |{image_pdf_id}| or Page id |{page_id}| != Image |{image_page_id}|"

            image_path = os.path.join(raw_image_folder, image_name)
            # page number starts with 1
            output_image_path = os.path.join(image_folder, f'{pdf_id}_{int(page_id) + 1}.png')
            # write the image with bounding boxes and numeric labels, labels at the top-left corner with -12 pixels to the left (outside the bbox for better visualization)
            bbox_ids = pages[page_id]['ordered_id']
            bboxes = [pages[page_id]['objects'][str(bid)]['bbox'] for bid in bbox_ids]
            assert bboxes == pages[page_id]['ordered_box'], f"Ordered bounding boxes are not consistent."
            label_position = lambda x, y, text: (-6 * len(text), 0)
            draw_image_with_bbox(image_path, bboxes, output_image_path, label_position=label_position)

            # update the page information
            tmp_page_info = {}
            tmp_page_info['page_path'] = output_image_path
            tmp_page_info['width'], tmp_page_info['height'] = pages[page_id]['width'], pages[page_id]['height']
            tmp_page_info['page_number'] = int(page_id) + 1 # starting from 1
            tmp_page_info['bbox'] = bboxes
            if pdf_id in errata and str(int(page_id) + 1) in errata[pdf_id]: # special care to pdf 28181161
                tmp_page_info['bbox_text'] = errata[pdf_id][str(int(page_id) + 1)]
            else:
                tmp_page_info['bbox_text'] = [pages[page_id]['objects'][str(bid)]['text'] for bid in bbox_ids]
            tmp_page_info['bbox_label'] = [pages[page_id]['objects'][str(bid)]['category_id'] for bid in bbox_ids]
            assert tmp_page_info['bbox_label'] == pages[page_id]['ordered_label'], f"Ordered labels are not consistent."

            # update the parent-child relations
            relations = set()
            for bid in pages[page_id]['objects']:
                obj = pages[page_id]['objects'][bid]['relations']
                mapped_bid1 = bbox_ids.index(str(bid))
                for type_, rel_bid in obj:
                    if type_ == 'parent':
                        mapped_bid2 = bbox_ids.index(str(rel_bid))
                        assert (mapped_bid2, mapped_bid1) not in relations, f"Reverse parent-child relation exists."
                        relations.add((mapped_bid1, mapped_bid2))
                    else:
                        assert type_ == 'child', f"Unknown relation type {type_}."
                        mapped_bid2 = bbox_ids.index(str(rel_bid))
                        assert (mapped_bid1, mapped_bid2) not in relations, f"Reverse parent-child relation exists."
                        relations.add((mapped_bid2, mapped_bid1))

            tmp_page_info['relations'] = sorted(relations)
            tmp_pdf_data['page_infos'].append(tmp_page_info)

    # preprocess the test data
    test_data = []
    def split_pdf_and_page(file_name):
        # e.g, 28789599.pdf_5.png
        pdf_id, page_number = file_name.split('.pdf_')
        return pdf_id, int(page_number.split('.')[0]) + 1

    # task a
    duplicate_set, duplicate_cnt = set(), 0
    taska = pd.read_csv(os.path.join(raw_data_folder, 'pdfvqa_taska_test_0509_clean.csv'))
    for row in taska.iterrows():
        row = row[1]
        if (row['file_name'], row['question'].strip()) in duplicate_set:
            duplicate_cnt += 1
            continue

        if isinstance(row['answer'], float) and math.isnan(row['answer']):
            continue

        duplicate_set.add((row['file_name'], row['question'].strip()))

        tmp_data = {}
        tmp_data['uuid'] = get_uuid(name=row['file_name'] + row['question'].strip() + 'taska')
        tmp_data['task_type'] = 'a'
        tmp_data['question'] = row['question'].strip()
        tmp_data['question_type'] = row['type']
        tmp_data['answer'] = row['answer']
        tmp_data['pdf_id'], tmp_data['page_number'] = split_pdf_and_page(row['file_name'])
        test_data.append(tmp_data)
    logger.info(f"Task A: remove {duplicate_cnt} duplicates from {len(taska)} examples.")

    # task b
    duplicate_set, duplicate_cnt = set(), 0
    taskb = pd.read_csv(os.path.join(raw_data_folder, 'pdfvqa_taskb_test_0503_clean.csv'))
    for row in taskb.iterrows():
        row = row[1]
        if (row['file'], row['question'].strip()) in duplicate_set:
            duplicate_cnt += 1
            continue

        if isinstance(row['answer'], float) and math.isnan(row['answer']):
            continue

        duplicate_set.add((row['file'], row['question'].strip()))

        tmp_data = {}
        tmp_data['uuid'] = get_uuid(name=row['file'] + row['question'].strip() + 'taskb')
        tmp_data['task_type'] = 'b'
        tmp_data['question'] = row['question'].strip()
        tmp_data['question_type'] = row['type']
        tmp_data['answer'] = row['answer']
        tmp_data['pdf_id'], tmp_data['page_number'] = split_pdf_and_page(row['file'])
        test_data.append(tmp_data)
    logger.info(f"Task B: remove {duplicate_cnt} duplicates from {len(taskb)} examples.")

    # task c
    duplicate_set, duplicate_cnt = set(), 0
    taskc = pickle.load(open(os.path.join(raw_data_folder, 'test_dataframe.pkl'), 'rb'))
    for row in taskc.iterrows():
        row = row[1] # row is a tuple (index, row)
        if (row['pmcid'], row['question'].strip()) in duplicate_set:
            duplicate_cnt += 1
            continue

        if isinstance(row['answer'], float) and math.isnan(row['answer']):
            continue

        duplicate_set.add((row['pmcid'], row['question'].strip()))

        tmp_data = {}
        tmp_data['uuid'] = get_uuid(name=row['pmcid'] + row['question'].strip() + 'taskc')
        tmp_data['task_type'] = 'c'
        tmp_data['question'] = row['question'].strip()
        tmp_data['question_type'] = row['question_type']
        tmp_data['answer'] = row['answer'] # List of str
        tmp_data['pdf_id'], tmp_data['page_number'] = row['pmcid'], None
        test_data.append(tmp_data)
    logger.info(f"Task C: remove {duplicate_cnt} duplicates from {len(taskc)} examples.")

    logger.info(f"In total, {len(pdf_data)} PDFs and {len(test_data)} examples are processed.")

    with open(os.path.join(processed_data_folder, test_data_name), 'w', encoding='UTF-8') as of:
        for data in test_data:
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open(os.path.join(processed_data_folder, pdf_data_name), 'w', encoding='UTF-8') as of:
        for data in pdf_data:
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
    return {'test_data': test_data, 'pdf_data': pdf_data}


def fuzzy_match_page(json_page_text: str, pdf_pages_text: List[str]) -> int:
    """Fuzzy match the JSON text content with the PDF pages and return the matched page number."""
    best_score = -1
    best_page = -1

    # Iterate over each page's text in the PDF
    for i, pdf_text in enumerate(pdf_pages_text):
        # Calculate the similarity score between JSON page text and the current PDF page text
        score = fuzz.ratio(json_page_text, pdf_text)
        if score > best_score:
            best_score = score
            best_page = i + 1  # Page number is 1-based

    return best_page


def get_page_width_and_height(pdf_path: str, page_number: int) -> Tuple[int, int]:
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    width = page.rect.width
    height = page.rect.height
    doc.close()
    return (width, height)


def process_tatdqa(
        raw_data_folder: str = 'data/dataset/tatdqa/raw_data',
        processed_data_folder: str = 'data/dataset/tatdqa/processed_data',
        test_data_name: str = 'test_data.jsonl',
        pdf_data_name: str = 'pdf_data.jsonl'
    ):

    """ Process the TATDQA dataset into a unified format and filter out questions on documents without complete original file
    @param:
        raw_data_folder: str, the path to the raw data folder.
        processed_data_folder: str, the path to the processed data folder.
        test_data_name: str, the name of the processed test data file, default is 'test_data.jsonl'.
        pdf_data_name: str, the name of the processed pdf data file, default is 'pdf_data.jsonl'.
    @return:
        test_data: List[Dict[str, Any]], return the list of processed data, each data point is a dictionary containing the following fields:
            {
                "uuid": "xxx-xxx-xxx-xxx", // str, UUID of the question
                "question": "What is the decrease in licensing revenue from Zyla (Oxaydo) from 2018 to 2019?", // str, the question text
                "question_type": "arithmetic", // str, chosen from [span, multi-span, arithmetic, count]
                "answer": [35,"thousand"] // List[Union[str,List[str], float]], the last element is scale(unit for answer of float type)
                "pdf_id": "xxx-xxx-xxx-xxx", // str, UUID of the PDF file
                }
        pdf_data: List[Dict[str, Any]], return the list of processed pdf files, each data point is a dictionary containing the following fields:
            {
                "pdf_id": "xxx-xxx-xxx-xxx", // str, UUID of the PDF file
                "num_pages": 9, // int, number of PDF pages
                "pdf_path": "data/dataset/tatdqa/raw_data/tat_docs/a10-networks-inc_2019.pdf",
                "page_infos": [ // List[Dict[str, Any]], information of already parsed page, which is a dictionary containing the following fields:
                    {
                        "page_number": 1, // int, the page number, starting from 1
                        "width": 640, // int, the width of the page
                        "height": 780, // int, the height of the page
                        "bbox": [ // List[Tuple[float, float, float, float]], [x0, y0, width, height]
                            [0, 0, 340, 230],
                            ...
                        ], // OCR bounding boxes in the current page, 4-tuple List
                        "bbox_text": [
                            "Text content of the first bbox.",
                            ...
                        ], // List[str], the OCR text content of the current page

                            ...
                        ] ,
                        "words": [
                            {
                                "word_list":[
                                    "Text content of the first word in the first bbox.",
                                    ...
                                ], // List[str]
                                "bbox_list":[
                                    [73, 73 ,108, 92],
                                    ...
                                ] // List[Tuple[float,float,float,float]], [x0, y0, width, height]
                            },
                            ... // other bbox
                        ] // List[Dict[str, Any]], the detailed word information of every bbox
                    },
                    ... // other pages
                ]
            }
    """

    # define unpressed data folder
    tatdqa_docs_folder = os.path.join(processed_data_folder, 'test')
    if not os.path.exists(tatdqa_docs_folder) or not os.path.isdir(tatdqa_docs_folder):
        with zipfile.ZipFile(os.path.join(raw_data_folder, 'tatdqa_docs_test.zip'), 'r') as zip_ref:
            zip_ref.extractall(processed_data_folder)
    tatdqa_pdf_folder = os.path.join(processed_data_folder, 'tat_docs')
    if not os.path.exists(tatdqa_pdf_folder) or not os.path.isdir(tatdqa_pdf_folder):
        with zipfile.ZipFile(os.path.join(raw_data_folder, 'tat_docs.zip'), 'r') as zip_ref:
            zip_ref.extractall(processed_data_folder)
    tatdqa_test_gold = os.path.join(raw_data_folder, 'tatdqa_dataset_test_gold.json')

    pdf_data = {}
    test_data = []

    # Load test question-answer data
    if not os.path.exists(tatdqa_test_gold):
        raise FileNotFoundError(f"File {tatdqa_test_gold} not found.")
    with open(tatdqa_test_gold, 'r') as f:
        tatdqa_test = json.load(f)


    def normalize_bbox(bbox, width_ratio, height_ratio):
        # original: [x0, y0, x1, y1]
        # after: [x0, y0, width, height]
        x0, y0, x1, y1 = bbox
        return [round(x0 * width_ratio, 1), round(y0 * height_ratio, 1), round((x1 - x0) * width_ratio, 1), round((y1 - y0) * height_ratio, 1)]


    # Process each document based on the question-answer dataset, tatdqa_dataset_test_gold.json
    # Note that, the same PDF may appear multiple times
    for qa in tqdm.tqdm(tatdqa_test):  #for one file
        doc_info = qa['doc']
        pdf_filename = doc_info['source']
        original_uid = doc_info['uid']
        pdf_id = get_uuid(name=pdf_filename) # reset uuid for each pdf file
        pdf_path = os.path.join(tatdqa_pdf_folder, pdf_filename)

        # Check if the PDF file exists, otherwise just skip the current test data group
        if os.path.exists(pdf_path):
            # Extract all the text from the PDF pages
            pdf_pages_text = get_pdf_page_text(pdf_path, generate_uuid=False)["page_contents"]

            # Construct the file path to the parsed JSON page
            json_filename = f"{original_uid}.json"
            json_filepath = os.path.join(tatdqa_docs_folder, json_filename)

            # initialize the item in pdf_data if not exists
            if pdf_id not in pdf_data:
                pdf_data_dict = {
                    "pdf_id": pdf_id,
                    "num_pages": len(pdf_pages_text),
                    "pdf_path": pdf_path,
                    "page_infos": []
                }
                pdf_data[pdf_id] = pdf_data_dict
            else: pdf_data_dict = pdf_data[pdf_id]

            # read parsed JSON page
            with open(json_filepath, 'r') as f:
                page_data = json.load(f)

            page_list, parsed_pages = [], [p['page_number'] for p in pdf_data_dict['page_infos']]
            for json_page in page_data['pages']:
                # Extract text from the JSON file for fuzzy matching
                json_page_text = '\n'.join([block['text'] for block in json_page['blocks']])

                # Fuzzy match the JSON page text with the original PDF text to get the page number
                matched_page_number = fuzzy_match_page(json_page_text, pdf_pages_text)
                page_list.append(matched_page_number)

                if matched_page_number in parsed_pages: # parsed before
                    continue

                width, height = get_page_width_and_height(pdf_path, matched_page_number)
                width_ratio, height_ratio = width / json_page['bbox'][2], height / json_page['bbox'][3]
                page_dict = {
                    "page_number": matched_page_number,
                    "width": width,  # real page width
                    "height": height,  # real page height
                    "bbox": [normalize_bbox(block['bbox'], width_ratio, height_ratio) for block in json_page['blocks']],
                    "bbox_text": [block['text'] for block in json_page['blocks']],
                    "words": [
                        {
                            "word_list": block['words']['word_list'],
                            "bbox_list": [normalize_bbox(bbox, width_ratio, height_ratio) for bbox in block['words']['bbox_list']]
                        } for block in json_page['blocks']
                    ]
                }

                pdf_data_dict['page_infos'].append(page_dict)

            questions = qa['questions']
            for question in questions:
                # create the json object
                test_data_dict = {
                    "uuid": get_uuid(name= pdf_filename + str(page_list) + question["question"].strip()), # reset uuid for each question
                    "question": question["question"],
                    "question_type": question["answer_type"],
                    "answer": [question["answer"], question["scale"]],
                    "pdf_id": pdf_id,
                    "page_number": page_list
                }
                test_data.append(test_data_dict)

    with open(os.path.join(processed_data_folder, test_data_name), 'w', encoding='UTF-8') as of:
        for data in test_data:
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
    pdf_data = list(pdf_data.values())
    for pdf in pdf_data:
        pdf['page_infos'] = sorted(pdf['page_infos'], key=lambda x: x['page_number'])
    with open(os.path.join(processed_data_folder, pdf_data_name), 'w', encoding='UTF-8') as of:
        for data in pdf_data:
            of.write(json.dumps(data, ensure_ascii=False) + '\n')

    # rename PDF files to ID
    tatdqa_test_docs_folder = os.path.join(processed_data_folder, 'test_docs')
    os.makedirs(tatdqa_test_docs_folder, exist_ok=True)
    for pdf in pdf_data:
        test_doc_path = os.path.join(tatdqa_test_docs_folder, f"{pdf['pdf_id']}.pdf")
        if not os.path.exists(test_doc_path):
            shutil.copy(pdf['pdf_path'], test_doc_path)

    return {'test_data': test_data, 'pdf_data': pdf_data}

def classify_question_type(data: Dict[str, Any], dataset: str) -> str:
    """ Classify the type for each question.
    @param:
        data: dict, the data extracted from test_data.jsonl
        dataset: str, the dataset name.
    """
    if dataset == 'pdfvqa':
        if data['question_type'] in ['existence', 'counting']:
            return data['question_type']
        if data['question_type'] in ['object_recognition', 'structural_understanding']:
            if data['answer'] == "No specific Section.":
                return f"{data['question_type']}_trivial"
            return data['question_type']
        if data['question_type'] in ['parent_relationship_understanding', 'child_relationship_understanding']:
            if isinstance(data['answer'], list) and (data['answer'][0] == 'No subsection!' or data['answer'][0] == 'No Section!'):
                return f"{data['question_type']}_trivial"
            return data['question_type']
        raise TypeError(f"Unknown question type {data['question_type']}.")
    elif dataset == 'tatdqa':
        return data['question_type']
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported for question type classification.")

def sampling_dataset(dataset: str = 'pdfvqa', sample_size: int = 300, output_file: str = None, random_seed: int = 2024):
    """ Sample the dataset for testing purposes.
    @param:
        dataset: str, the dataset name.
        sample_size: int, the sample size for the dataset.
        output_file: str, the output file name for the sampling .jsonl file.
    """
    dataset_path = os.path.join(DATASET_DIR, dataset, 'processed_data', 'test_data.jsonl')
    with open(dataset_path, 'r', encoding='UTF-8') as inf:
        data = [json.loads(line) for line in inf]
    typed_data = {}
    for d in data:
        question_type = classify_question_type(d, dataset)
        if question_type not in typed_data:
            typed_data[question_type] = []
        typed_data[question_type].append(d)
    # for different question types, sample the data in proportion
    for k, v in typed_data.items():
        logger.info(f"type = {k}, size = {len(v)}")
    typed_sample_size = {tp: math.ceil(sample_size * 1.0 * len(typed_data[tp]) / len(data)) for tp in typed_data}
    sampled_data = []
    random.seed(random_seed)
    for tp in typed_data:
        if len(typed_data[tp]) <= typed_sample_size[tp]:
            typed_sample_size[tp] = len(typed_data[tp])
            sampled_data.extend(typed_data[tp])
        else:
            sampled_data.extend(random.sample(typed_data[tp], typed_sample_size[tp]))
        logger.info(f'Sample {typed_sample_size[tp]} test data for type {tp}.')

    sample_size = len(sampled_data)
    output_path = os.path.join(DATASET_DIR, dataset, 'processed_data', output_file) if output_file is not None else dataset_path.replace('test_data.jsonl', f'test_data_sample_{sample_size}.jsonl')
    with open(output_path, 'w', encoding='UTF-8') as of:
        for d in sampled_data:
            of.write(json.dumps(d, ensure_ascii=False) + '\n')
        logger.info(f"Sampled {sample_size} test data saved to {output_path} for dataset {dataset}.")
    return sampled_data


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Dataset relevant utilities.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--function', type=str, default='preprocess', choices=['preprocess', 'sampling'], help='Function name.')
    parser.add_argument('--sample_size', type=int, default=300, help='Sample size for the dataset.')
    parser.add_argument('--output_file', type=str, help='Output file name of the sampling .jsonl file.')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed for sampling.')
    args = parser.parse_args()

    FUNCTIONS = {
        'preprocess': {
            'pdfvqa': process_pdfvqa,
            'tatdqa': process_tatdqa
        },
        'sampling': sampling_dataset,
    }
    if args.function == 'preprocess':
        FUNCTIONS[args.function][args.dataset]()
    elif args.function == 'sampling':
        FUNCTIONS[args.function](args.dataset, sample_size=args.sample_size, output_file=args.output_file, random_seed=args.random_seed)
    else:
        raise ValueError(f"Function {args.function} not supported for dataset {args.dataset}.")
#coding=utf8
import os, json, sys, logging, re, zipfile, pickle
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Union, Optional, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions.common_functions import get_uuid
from utils.functions.image_functions import draw_image_with_bbox


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
                        ], // List[int], labels for each bbox, 1->main text, 2->section title, 3->
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

    # preprocess images of each PDF page
    for pdf_id in docs:
        tmp_pdf_data = {}
        tmp_pdf_data['pdf_id'] = pdf_id
        # special care to pdf with id "28649313", missing the first 16 pages
        tmp_pdf_data['num_pages'] = len(docs[pdf_id]['pages']) if pdf_id != '28649313' else 21
        tmp_pdf_data['page_infos'] = [] if pdf_id != '28649313' else [{} for _ in range(16)]
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
            draw_image_with_bbox(image_path, bboxes, output_image_path, label_position=(-8, 0))

            # update the page information
            tmp_page_info = {}
            tmp_page_info['page_path'] = output_image_path
            tmp_page_info['width'], tmp_page_info['height'] = pages[page_id]['width'], pages[page_id]['height']
            tmp_page_info['page_number'] = int(page_id) + 1 # starting from 1
            tmp_page_info['bbox'] = bboxes
            tmp_page_info['bbox_text'] = [pages[page_id]['objects'][str(bid)]['text'] if 'text' in pages[page_id]['objects'][str(bid)] else None for bid in bbox_ids] # special care to pdf 28181161, no text
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

    with open(os.path.join(processed_data_folder, test_data_name), 'w') as of:
        for data in test_data:
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open(os.path.join(processed_data_folder, pdf_data_name), 'w') as of:
        for data in pdf_data:
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
    return {'test_data': test_data, 'pdf_data': pdf_data}


def process_tatdqa(
        raw_data_folder: str = 'data/dataset/tatdqa/raw_data',
        processed_data_folder: str = 'data/dataset/tatdqa/processed_data'
    ):
    """ Process the TATDQA dataset.
    @param:
        raw_data_folder: str, the path to the raw data folder.
    """
    pass


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Dataset relevant utilities.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    args = parser.parse_args()

    if args.dataset == 'pdfvqa':
        process_pdfvqa()
    elif args.dataset == 'tatdqa':
        process_tatdqa()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
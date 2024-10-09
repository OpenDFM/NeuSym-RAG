# Datasets

We choose and pre-process the following benchmarks for experiments:
- [`pdfvqa`](../data/dataset/pdfvqa/) about biology paper: [PDF-VQA: A New Dataset for Real-World VQA on PDF Documents](https://arxiv.org/pdf/2304.06447)
- [`tatdqa`](../data/dataset/tatdqa/) about financial report: [TAT-DQA: Towards Complex Document Understanding By Discrete Reasoning](https://arxiv.org/pdf/2207.11871)

## Folder Structure

```txt
- data/dataset/
    - pdfvqa/
        - raw_data/ # raw dataset
            - pdfvqa_taska_test_0509_clean.csv
            - pdfvqa_taskb_test_0503_clean.csv
            - test_dataframe.pkl # indeed, task c
            - test_images.zip
            - test_doc_info_visual.pkl # layout structure information
        - processed_data/ # pre-processed dataset
    - tatdqa/
        - raw_data/ # raw dataset
            - tat_qa.csv # QA in TatHybrid of dataset UDA
            - tatdqa_dataset_test_gold.json # QA in dataset tatdqa
            - tat_docs.zip # source PDF files
        - processed_data/ #pre-processed dataset
            
```


## Downloading

For PDF-VQA, please download them from the following links into folder `data/dataset/pdfvqa/raw_data/`:
- [pdfvqa_taska_test_0509_clean.csv](https://drive.google.com/file/d/1gGIzsSZHVokehACx7h-SOk5K1uEEXNpq/view?usp=drive_link)
- [pdfvqa_taskb_test_0503_clean.csv](https://drive.google.com/file/d/1FrAB0tKcVg3r67yi-Q2pDw928ACxeRzH/view?usp=drive_link)
- [test_dataframe.pkl](https://drive.google.com/file/d/1-F242FFvubAIpjXPItFc_eGUs3dzb6QO/view?usp=drive_link)
- [test_images.zip](https://drive.google.com/drive/folders/1A2cI3uJUU_1ZliOKpHmYa07VfvZCwOo1?usp=drive_link)
- [test_doc_info_visual.pkl](https://drive.google.com/file/d/1knSVmocw4-_FF98bFMdVSvhnUn3mPUvm/view?usp=drive_link)

> Notice, it seems that the [official repository](https://github.com/adlnlp/pdfvqa?tab=readme-ov-file) swapped the download links for the validation set ([pdfvqa_taska_val_0509_clean.csv](https://drive.google.com/file/d/1HIYxpGCcXdQo42b79Eqfmji9U9-YmUjp/view?usp=drive_link))and test set ([pdfvqa_taska_test_0509_clean.csv](https://drive.google.com/file/d/1gGIzsSZHVokehACx7h-SOk5K1uEEXNpq/view?usp=drive_link)) of task A. We follow the file name to choose `pdfvqa_taska_test_0509_clean.csv` as the test set instead of `README.md`.

For TAT-DQA, please download them from the following links into folder `data/dataset/tatdqa/raw_data/`:
- [tatdqa_dataset_test_gold.json](https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2)
- [tatdqa_docs_test.zip](https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2)
- [tat_docs.zip](https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/main/src_doc_files/tat_docs.zip?download=true)


> Notice, the original dataset only contains the specific or oracle PDF page instead of the complete PDF file for each instance, which we believe is not practical in real-world scenarios. Thus, we refer to [UDA-Benchmark](https://github.com/qinchuanhui/UDA-Benchmark?tab=readme-ov-file#book-dataset-uda-qa) and download the raw PDF files ([tat_docs.zip](https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/main/src_doc_files/tat_docs.zip?download=true)).

TODO: other benchmarks relevant to PDF.

## Pre-processing

### PDF-VQA
We preprocess each example into folder `data/dataset/pdfvqa/processed_data/`. The entire dataset is splitted into 3 parts:
- [bbox_images](../data/dataset/pdfvqa/processed_data/bbox_images): which contains images for each page (e.g., `15450119_1.png`) in one PDF with bounding boxes drawed.
- [test_data.jsonl](../data/dataset/pdfvqa/processed_data/test_data.jsonl): JSON line file, each test example is represented with one JSON dict containing the following fields:
```json
{
    "uuid": "xxx-xxx-xxx-xxx", // str, UUID of the test example
    // "task_type": "a", // str, chosen from a, b, c
    "question": "Is it correct that there is no figure on the top left?",
    "question_type": "existence", // str, chosen from ['existence', 'counting', 'object_recognition', 'structural_understanding', 'parent_relationship_understanding', 'child_relationship_understanding']
    "answer": true, // Union[bool, str, List[str]], three types of answers for different task types
    "pdf_id": "15450119", // str, id of the PDF
    "page_number": 3 // int, reference page number for this question, starting from 1. Note that this field is None for task c
}
```
- [pdf_data.jsonl](../data/dataset/pdfvqa/processed_data/pdf_data.jsonl): JSON line file, each PDF file is represented with one JSON dict.
```json
{
    "pdf_id": "15450119", // str, id of the PDF file
    "num_pages": 9, // int, total number of PDF pages
    "page_infos": [ // List[Dict[str, Any]], information of each page
        {
            "page_number": 1, // int, the page number, starting from 1
            "page_path": "data/dataset/pdfvqa/processed_data/bbox_images/15450119_1.png",
            "width": 640, // width of the page image
            "height": 780, // height of the page image
            "bbox": [ // List[Tuple[float, float, float, float]], [x0, y0, width, height]
                [32, 120, 612, 242],
                ...
            ], // OCR bounding boxes in the current page, 4-tuple List
            "bbox_text": [
                "Text content of the first bbox.",
                ...
            ], // List[str], OCR text content of each bbox, if not recognized, None
            "bbox_label": [
                1,
                ...                            
            ], // List[int], labels for each bbox, 1->main text, 2->section title, 3->list such as references, 4->tables, 5->figures
            "relations": [ // List[Tuple[int, int]], parent-child relations between bounding boxes
                [0, 1], // numbers represent bbox index, starting from 0
                ... 
            ] // the first element is the parent, and the second is the child
        },
        ... // other pages
    ]
}
```
> Special attention to PDFs with id `28649313` and `28181161`. PDF `28649313` missing the first 16 pages, only start from page 17. And PDF `28181161` missing text content for many bounding boxes in page 8-11.

#### Running script:
```sh
python utils/dataset_utils.py --dataset pdfvqa
```

### TAT-DQA
We preprocess each example into folder `data/dataset/tatdqa/processed_data/`. The entire dataset is processed as:
- [test_data.jsonl](../data/dataset/tatdqa/processed_data/test_data.jsonl): JSON line file, each test example is represented with one JSON dict containing the following fields:
```json
{
    "pdf_id": "xxxx-xxxx-xxxx", // str, name of the PDF document
    "uuid": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // str, UID of the question
    "question": "What is the decrease in licensing revenue from Zyla (Oxaydo) from 2018 to 2019?", // str, the question text
    "question_type": "arithmetic", // str, chosen from [span, multi-span, arithmetic, count]
    "answer": [35, "thousand"], // List[Union[str,List[str], float]],the last element is scale(unit for answer of float type)
}
```
- [pdf_data.jsonl](../data/dataset/tatdqa/processed_data/pdf_data.jsonl): JSON line file, each PDF file is represented with one JSON dict.
```json
{
    "pdf_id": "xxx-xxx-xxx-xxx", // str, UUID of the PDF file
    "num_pages": 121, // int, number of PDF pages
    "pdf_path": "data/dataset/tatdqa/raw_data/tat_docs/a10-networks-inc_2019.pdf",
    "page_infos": [ // List[Dict[str, Any]], information of already parsed page, which is a dictionary containing the following fields:
        {
            "page_number": 23, // int, the page number, starting from 1
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
                    
            "words": [
                {
                "word_list":[ 
                    "Text content of the first word in the first bbox.",
                    ...
                ], // List[str]
                "bbox_list":[ 
                    [73, 73 ,108, 92],
                    ...
                ] // List[Tuple[float,float,float,float]], [x0, yo, width, height]
                },
                ... // other bbox
            ] // List[Dict[str, Any]], the detailed word information of every bbox
        },
                    ... // other pages
    ]
}
```

```
def evaluate(pred: str, gold: data['answer'], question_type: enum) -> float:
    if question_type == '?':
        eval_func1()
        pass
    elif ...
        eval_func2()

#### Running script:
```sh
python utils/dataset_utils.py --dataset tatdqa
```

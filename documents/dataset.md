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
        - raw_data/ # raw dataset, documents, images
            
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
- [tatdqa_dataset_test_gold.json, tatdqa_docs_test.zip](https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2)
- [tat_docs.zip](https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/main/src_doc_files/tat_docs.zip?download=true)

> Notice, the original dataset only contains the specific or oracle PDF page instead of the complete PDF file for each instance, which we believe is not practical in real-world scenarios. Thus, we refer to [UDA-Benchmark](https://github.com/qinchuanhui/UDA-Benchmark?tab=readme-ov-file#book-dataset-uda-qa) and download the raw PDF files ([tat_docs.zip](https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/main/src_doc_files/tat_docs.zip?download=true)).

TODO: other benchmarks relevant to PDF.

## Pre-processing

### PDF-VQA
We preprocess each example into folder `data/dataset/pdfvqa/processed_data/`. The entire dataset is splitted into 3 parts:
- [bbox_images](../data/dataset/pdfvqa/bbox_images): which contains
- test_data.jsonl:
- pdf_data.jsonl:
```txt

```
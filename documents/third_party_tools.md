## Third Party Tools Guideline

This document describes the detailed installation, use case, and useful links to third party tools that may be utilized in this project, especially during PDF/image parsing.

### Processing Logic of `get_ai_research_metadata`

Core input argument: `pdf_path`. It can take the following 4 forms:
- **uuid string**:
    - In this case, the metadata dict should be pre-processed and stored in `data/dataset/airqa/uuid2papers.json`, directly return that dict.
    - If not found in `uuid2papers.json`, raise ValueError.
- **local pdf path**:
    - Case 1: `data/dataset/airqa/papers/subfolder/{uuid}.pdf`. We will directly extract the UUID and reduce to situation **uuid string**.
    - Case 2: `/path/to/any/folder/anyfilename.pdf`. Firstly, we assume the paper title MUST occur in top lines of the first page. We use LLM to get the paper title from these texts. Then, we resorts to scholar API to extract the paper metadata. After processing, the original local file will be moved and renamed to the field `pdf_path` in the metadata dict.
- **remote URL starts with http**
    - In this case, we will firstly download the PDF file to `TMP_DIR` (by default, ./tmp/). Then, it degenerates to situation **local pdf path** case 2.
    - Similarly, after processing, the downloaded local PDF file will be moved and renamed to the field `pdf_path` in the metadata dict.
- **paper title**:
    - In this case, we will directly call scholar APIs to obtain the metadata.
    - After getting the metadata, we will also download and rename the PDF file according to fields `pdf_url` and `pdf_path` in the metadata dict.
Attention that, after calling `get_ai_research_metadata`, new paper UUID will be added into `data/dataset/airqa/uuid2papers.json` by default. If you want to prohibit the writing operation, add keyword argument 


### DBLP Scholar API

- No extra libs needed, `requests` + `urllib` + `bs4` is enough
- Code snippets:
```py
from utils.functions.ai_research_metadata import dblp_scholar_api

print("Obtained paper is:\n", dblp_scholar_api(
    title="Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows?",
    limit=10, # restrict the maximum number of hits by DBLP API
    threshold=95, # DBLP search uses very weak matching criterion, we use fuzz.ratio to re-order the results ( only ratio score > threshold will be maintained )
    allow_arxiv=True # by default, False, since we implement another arxiv scholar API, but can be changed to True, such that arxiv version of papers will not be ignored
))
```

### More Scholar APIs like semantic-scholar and arxiv (TODO)



### Unstructured

- Installation: [Github link](https://github.com/Unstructured-IO/unstructured?tab=readme-ov-file#installing-the-library)
    - For MacOS: suggest using PyPi
        - brew install libmagic poppler tesseract
        - pip install "unstructured[all-docs]"
    - For Windows:
        - pip install "unstructured[all-docs]"
        - [tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
        - [poppler](https://github.com/oschwartz10612/poppler-windows/releases/): After extraction, add the `bin/` directory to your [PATH](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).
- Core functions: [`partition_pdf`](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pdf) and [`partition_image`](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-image)
- For `partition_pdf`, actually it converts the raw PDF into images, keyword parameters include:
    - `languages: List[str]`, e.g., `languages=["eng"]`, see [Tesseract](https://github.com/tesseract-ocr/tessdata) for all choices
    - `strategy: str`, this parameter controls the processing method, including:
        - `hi_res`: if `infer_table_structure=True` or `extract_images_in_pdf=True`, use this strategy
        - `ocr_only`: if text is not extractable, this strategy uses Tesseract for OCR
        - `fast`: the default strategy if text can be extracted with package `pdfminer`
    - `infer_table_structure: bool`, used to identify tables in PDF
    - `extract_images_in_pdf: bool`, often used in combination with `extract_image_block_types=['Image', 'Table']`, `extract_image_block_to_payload: bool`, and `extract_image_block_output_dir: str`
        - all available block/element types can be found in class `ElementType` under `unstructured.documents.elements`. Namely,
```python
['Abstract', 'Address', 'BulletedText', 'Caption', 'Checked', 'CheckBoxChecked', 'CheckBoxUnchecked', 'CodeSnippet', 'CompositeElement', 'EmailAddress', 'Field-Name', 'Figure', 'FigureCaption', 'Footer', 'Footnote', 'Form', 'Formula', 'FormKeysValues', 'Header', 'Headline', 'Image', 'Link', 'List', 'ListItem', 'List-item', 'NarrativeText', 'PageBreak', 'Page-footer', 'Page-header', 'PageNumber', 'Paragraph', 'Picture', 'RadioButtonChecked', 'RadioButtonUnchecked', 'Section-header', 'Subheadline', 'Table', 'Text', 'Threading', 'Title', 'UncategorizedText', 'Unchecked', 'Value']
```
    - `max_partition: Optional[int]`, please set it to `None` and disable the default value `1500`
- For `partition_image`, the parameters are almost the same with `partition_pdf`, and the source code can be checked in [source code](https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py)

#### Use Case

1. Extract images/tables figures from PDF file:
```python
import os
from unstructured.partition.pdf import partition_pdf
# downloading inference models may need to set the proxy
os.environ['http_proxy'] = 'http://127.0.0.1:58591'
os.environ['https_proxy'] = 'http://127.0.0.1:58591'

# setting the padding size to enlarge the region of the extracted images
os.environ['EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD'] = '5'
os.environ['EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD'] = '10'
os.makedirs("images", exist_ok=True)
elements = partition_pdf("./paper.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,
    extract_image_block_types=['Image', 'Table'], # all other elements will use 'figure' as the saved file basename
    extract_image_block_output_dir='images',
    max_partition=None
)
```
And the output under the specified `images/` folder should be like:
```txt
- images/: # ${basename}-${page-num}-${index}.jpg
    - figure-1-1.jpg
    - figure-1-2.jpg
    - figure-2-3.jpg
    - ...
    - table-2-1.jpg
    - table-3-2.jpg
    - table-4-3.jpg
    - table-4-4.jpg
```
2. Extract and format tables in PDF file:
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(filename='./paper.pdf',
                         infer_table_structure=True,
                         strategy='hi_res',
                         max_partition=None)

tables = [el for el in elements if el.category == "Table"]

print(tables[0].text)
print(tables[0].metadata.text_as_html)
```
Example html output:
```txt
<table><thead><tr><th>Benchmark</th><th>Field</th><th>Exec. Env?</th><th>Ent. Serv.?.</th><th>GUI Support?</th><th>#Apps/ Sites</th><th>= # Exec.-based Eval. Func.</th><th># Tasks</th></tr></thead><tbody><tr><td></td><td>Text-to-SQL</td><td>xK</td><td>xK</td><td>xK</td><td>1</td><td>0</td><td>1034</td></tr><tr><td></td><td>Data Science</td><td>K</td><td>x</td><td>K</td><td>1</td><td>0</td><td>1000</td></tr><tr><td></td><td>Data Science</td><td>K</td><td>x</td><td>K</td><td>1</td><td>0</td><td>1082</td></tr><tr><td>MLAgentBench</td><td>Machine Learning</td><td>v</td><td>xK</td><td>xK</td><td>4</td><td>13</td><td>13</td></tr><tr><td>SWE-Bench</td><td>Software Engineering</td><td>xK</td><td>xK</td><td>xK</td><td>2</td><td>1</td><td>2294</td></tr><tr><td>Mind2Web</td><td>Web</td><td>x</td><td>xK</td><td>v</td><td>137</td><td>0</td><td>2000</td></tr><tr><td>WEBLINX</td><td>Web</td><td>xK</td><td>x</td><td>v</td><td>155</td><td>0</td><td>2337</td></tr><tr><td>WorkArena</td><td>Web</td><td>v</td><td>v</td><td>v</td><td>1</td><td>7</td><td>29</td></tr><tr><td>AndroidWorld [25</td><td>Android</td><td>v</td><td>x</td><td>v</td><td>20</td><td>6</td><td>116</td></tr><tr><td>WebArena</td><td>Web</td><td>v</td><td>x</td><td>v</td><td>5</td><td>5</td><td>812</td></tr><tr><td>OSWorld</td><td>Computer Control</td><td>v</td><td>xK</td><td>v</td><td>9</td><td>134</td><td>369</td></tr><tr><td>Spider2-V</td><td>Data Science &amp; Engineering w/ Computer Control</td><td>7</td><td>7</td><td>7</td><td>30</td><td>151</td><td>494</td></tr></tbody></table>
```

### MinerU

To be done.
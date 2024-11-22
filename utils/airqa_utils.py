""" This module contains utility functions for the AIR-QA dataset.
"""
import requests, shutil, subprocess
import tqdm, fitz
import os, sys, re, json, logging
from typing import Dict, Any, Optional, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions.common_functions import get_uuid
from agents.models import get_single_instance
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


AIRQA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'dataset', 'airqa'
)


abbrev_mappings = {
    "conll": "CoNLL",
    "semeval": "SemEval",
    "neurips": "NeurIPS",
    "corl": "CoRL",
    "interspeech": "Interspeech",
    "recsys": "RecSys",
    "automl": "AutoML",
    "collas": "CoLLAs"
}


def get_airqa_paper_uuid(title: str, conference_year: str = 'uncategorized') -> str:
    """ Get the UUID of a paper in the AIR-QA dataset.
    `meta` is lowercase {conference}{year}, e.g., 'acl2024'.
    """
    # normalize the paper title
    paper = title.strip() + '-' + conference_year.lower()
    return get_uuid(paper, uuid_type='uuid5', uuid_namespace='dns')


def get_all_used_paper_uuids(
        example_dir: str = os.path.join(AIRQA_DIR, 'examples'),
        output_file: str = os.path.join(AIRQA_DIR, 'used_uuids.json')
    ) -> List[str]:
    """ Extract all used paper UUIDs from the AIR-QA examples.
    """
    uuids = set()
    for file in os.listdir(example_dir):
        if file.endswith('.json'):
            with open(os.path.join(example_dir, file), 'r', encoding='utf8') as inf:
                data = json.load(inf)
                uuids.update(data['pdf_id'])
    uuids = list(uuids)
    if output_file is not None:
        with open(output_file, 'w', encoding='utf8') as ouf:
            json.dump(uuids, ouf, ensure_ascii=False, indent=4)
    return uuids


def get_answer_from_llm(question_uuid: Optional[str] = None, question: Optional[str] = None, add_answer_format: bool = True, model: str = 'gpt-4o', temperature: float = 0.7, top_p: float = 0.95) -> str:
    """ Get the answer from the LLM model.
    @param:
        question_uuid: str, the UUID of the question
        model: str, the LLM model name
        temperature: float, the temperature parameter
        top_p: float, the top-p parameter
    @return:
        str, the answer/response from the LLM model
    """
    assert question_uuid is not None or question is not None, "Either question_uuid or question must be provided."
    if question_uuid is not None:
        example = os.path.join('data', 'dataset', 'airqa', 'examples', question_uuid + '.json')
        with open(example, 'r', encoding='utf8') as inf:
            data = json.load(inf)
            question = data['question'] + ' ' + data['answer_format'] if add_answer_format else data['question']
    client = get_single_instance(model_name=model)
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent agent who is expert in artificial intelligence. Please answer the following question with respect to AI research papers."
        },
        {
            "role": "user",
            "content": f"Here is the question: {question}\nLet's think step by step."
        }
    ]
    response = client.get_response(messages, model=model, temperature=temperature, top_p=top_p)
    logger.info(f"Response from LLM model: {response}")
    return response


def make_airqa_dataset(airqa_dir: str = AIRQA_DIR):
    output_path = os.path.join(airqa_dir, 'test_data.jsonl')
    indir = os.path.join(airqa_dir, 'examples')
    json_files = os.listdir(indir)
    count = 0
    with open(output_path, 'w', encoding='utf8') as of:
        for fp in json_files:
            fp = os.path.join(indir, fp)
            with open(fp, 'r', encoding='utf8') as inf:
                data = json.load(inf)
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1
    logger.info(f"Merge {count} AIR-QA examples into {output_path}.")
    return output_path


def download_html(url: str, html_path: str = None):
    response = requests.get(url)
    if response.status_code == 200:
        if html_path is None:
            return response.text
        with open(html_path, 'w', encoding='utf8') as ouf:
            ouf.write(response.text)
        return html_path
    else:
        logger.error(f"Failed to download HTML from {url}.")
        return None


def download_paper_pdf(pdf_url: str, pdf_path: str) -> Optional[str]:
    """ Download the PDF file from the `pdf_url` into `pdf_path`. Just return the relative `pdf_path` if succeeded.
    """
    if os.path.exists(pdf_path) and os.path.isfile(pdf_path): # PDF file already exists
        logger.warning(f"PDF file {pdf_path} already exists. Just ignore the download from {pdf_url}.")
        return pdf_path
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"Downloaded paper `{pdf_url}` successfully to: {pdf_path}")
            return pdf_path
        else:
            logger.error(f"Failed to download paper `{pdf_url}`. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while downloading PDF file: {e}")
    return None


def get_relative_path(file_path: str) -> str:
    """ Get the relative path of a file.
    """
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.relpath(os.path.abspath(file_path), working_dir)


def repair_pdf_with_qpdf(pdf_path: str) -> str:
    if shutil.which("qpdf") is not None:
        subprocess.run(["qpdf", pdf_path, pdf_path + ".repaired.pdf"])
        subprocess.run(["mv", pdf_path + ".repaired.pdf", pdf_path])
    else:
        logger.error(f"[Error]: Try to use `qpdf` to repair the PDF file {pdf_path}, but `qpdf` is not installed.")
    return pdf_path


def get_num_pages(pdf_path: str) -> int:
    """ Get the number of pages in a PDF file.
    """
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    doc.close()
    if num_pages == 0:
        repair_pdf_with_qpdf(pdf_path)
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    return num_pages


def crawl_acl_anthology_papers(
        url: str = "https://aclanthology.org/events/acl-2024/",
        output_dir: str = AIRQA_DIR,
        errata_file: Optional[str] = os.path.join(AIRQA_DIR, 'errata.json')):
    """ Crawl papers from ACL Anthology website. Use `get_airqa_paper_uuid` to generate UUIDs and rename the output PDFs under `output_dir`. Save meta data and UUID -> paper mappings into uuid2papers.json.
    @param:
        url: str, the URL of the ACL Anthology. This parameter will be used to download the HTML file into `html_path`.
        output_dir: str, the directory to save the PDF files.
        errata_file: str, the file path to the errata JSON file.
    """
    # get UUID -> paper mappings
    uuid2papers_path = os.path.join(output_dir, f'uuid2papers.json')
    if os.path.exists(uuid2papers_path):
        with open(uuid2papers_path, 'r', encoding='utf8') as inf:
            uuid2papers = json.load(inf)
    else: uuid2papers = {}

    # errata file, maybe deleted in future
    if os.path.exists(errata_file):
        with open(errata_file, 'r', encoding='utf8') as inf:
            errata = json.load(inf)
    else: errata = {}

    download_html(url, 'index.html')
    with open('index.html', 'r', encoding='utf8') as inf:
        html_doc = inf.read()
    if os.path.exists('index.html'):
        os.remove('index.html')

    soup = BeautifulSoup(html_doc, 'html.parser')
    conference_full = soup.select_one('h2#title').get_text().strip() # e.g., Annual Meeting of the Association for Computational Linguistics (2024)
    conference, year = os.path.basename(url.rstrip('#').rstrip(os.sep)).lower().split('-')
    paper_dir = os.path.join(output_dir, 'papers', conference + year) # folder to save all paper PDFs


    def parse_acl_paper_meta(node):
        # node is a <p> tag
        meta = {
            "uuid": '', # UUID generated by the title
            "title": '', # paper title
            "conference_full": conference_full, # full title of the conference
            "conference": abbrev_mappings.get(conference, conference.upper()), # conference abbreviation is ACL/EMNLP/NAACL
            "year": int(year), # conference year
            "volume": volume_title, # volume title
            "bibtex": '', # bibtex citation
            "authors": [], # authors list
            "pdf_url": '', # URL to download the PDF
            "pdf_path": '', # local path to save the PDF, rename it with the UUID
            "num_pages": -1, # number of pages in the PDF
            "abstract": None # paper abstract, will be assigned in another function
        }
        for idx, span in enumerate(node.children):
            if span.name != 'span': continue
            if idx == 0:
                # get pdf_url and bib
                link_nodes = span.find_all('a')
                for link in link_nodes:
                    href = link.get('href')
                    if not href: continue
                    if not href.startswith('http'): # get full URL
                        href = 'https://aclanthology.org/' + href.lstrip('/')
                    if href.endswith('.pdf'): # pdf_url
                        meta['pdf_url'] = href
                    elif href.endswith('.bib'): # get raw bib text
                        meta['bibtex'] = download_html(href)
                    else: continue # ignore other links
            else:
                # get title, authors
                meta['title'] = span.select_one('strong > a').get_text().strip()
                meta['uuid'] = get_airqa_paper_uuid(meta['title'], meta['conference'] + str(meta['year']))
                download_path = os.path.join(paper_dir, meta['uuid'] + '.pdf')
                if not download_paper_pdf(meta['pdf_url'], download_path):
                    logger.error(f'Failed to download the PDF file from {meta["pdf_url"]} into {download_path}.')

                meta['num_pages'] = get_num_pages(download_path)
                next_sibling = node.find_next_sibling()
                if next_sibling is not None and next_sibling.name == 'div':
                    meta['abstract'] = next_sibling.get_text().strip()
                else: # try parse the abstract from first page, content between "\nAbstract\n" and "\n1\n"
                    doc = fitz.open(download_path)
                    page_content = doc[0].get_text()
                    doc.close()
                    abstract_content = re.search(r"\nAbstract\n(.*?)\n1\n", page_content, flags=re.DOTALL)
                    if abstract_content:
                        abstract_content = abstract_content.group(1).strip()
                        abstract = ""
                        for line in abstract_content.split('\n'):
                            if abstract.endswith('-'):
                                abstract = abstract[:-1] + line.strip()
                            else:
                                abstract += ' ' + line.strip()
                        meta['abstract'] = abstract.strip()

                meta['pdf_path'] = get_relative_path(download_path)
                for author in span.select('a[href^="/people/"]'):
                    meta['authors'].append(author.get_text().strip())
        return meta


    for volume in soup.select('section#main h4 > a.align-middle'):
        volume_title = volume.get_text().strip()
        parent_node = volume.parent.parent
        skip_first_p = False # the first element is not a paper, more like an official summary or introduction
        logger.info(f"Processing volume: {volume_title}")
        for child in tqdm.tqdm(parent_node.find_all(recursive=False)):
            if child.name == 'p':
                if skip_first_p:
                    meta = parse_acl_paper_meta(child)
                    if meta['uuid'] in uuid2papers:
                        logger.warning(f"UUID {meta['uuid']} already exists in the UUID -> paper mappings {meta['title']}.")
                        continue
                    uuid2papers[meta['uuid']] = meta
                else: skip_first_p = True

        # save the UUID -> paper mappings after finishing each volume
        with open(uuid2papers_path, 'w', encoding='utf8') as ouf:
            json.dump(uuid2papers, ouf, ensure_ascii=False, indent=4)
    return uuid2papers


if __name__ == '__main__':

    for url in ['https://aclanthology.org/events/acl-2023/', 'https://aclanthology.org/events/acl-2024/']:
        crawl_acl_anthology_papers(url, output_dir=AIRQA_DIR)
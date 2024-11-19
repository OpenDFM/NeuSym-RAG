""" This module contains utility functions for the AIR-QA dataset.
"""
import requests
import tqdm
import os, sys, re, json, logging
from typing import Dict, Any, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions.common_functions import get_uuid
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


def get_airqa_paper_uuid(title: str, meta: str = 'acl2024') -> str:
    """ Get the UUID of a paper in the AIR-QA dataset.
    `meta` is lowercase {conference}{year}, e.g., 'acl2024'.
    """
    # normalize the paper title
    paper = title.strip() + '-' + meta.lower()
    return get_uuid(paper, uuid_type='uuid5', uuid_namespace='dns')


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


def download_and_rename_pdf(metadata: Dict[str, Any], paper_dir: str = 'data/dataset/airqa/papers/acl2024'):
    """ Download the PDF file from the URL in the metadata and rename it with the UUID.
    Note that, use relative path instead of absolute path in the metadata.
    """
    pdf_path = os.path.join(paper_dir, str(metadata['uuid']) + '.pdf')
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    relative_path = os.path.relpath(os.path.abspath(pdf_path), working_dir)
    if os.path.exists(pdf_path): # PDF file already exists
        metadata['pdf_path'] = relative_path
        return pdf_path

    pdf_url = metadata['pdf_url']
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.debug(f"Downloaded paper `{metadata['title']}` successfully to: {pdf_path}")
        else:
            logger.error(f"Failed to download paper `{metadata['title']}`. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while downloading PDF file: {e}")
    metadata['pdf_path'] = relative_path
    return pdf_path


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

    # errata file
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
                title = span.select_one('strong > a').get_text().strip()
                meta['uuid'] = get_airqa_paper_uuid(title, meta['conference'] + str(meta['year']))
                meta['title'] = title
                for author in span.select('a[href^="/people/"]'):
                    meta['authors'].append(author.get_text().strip())
        return meta

    def parse_acl_paper_abstract(node):
        # node is a <div> tag
        return node.get_text().strip() if node is not None and node.name == 'div' else None
    start = False
    for volume in soup.select('section#main h4 > a.align-middle'):
        volume_title = volume.get_text().strip()
        parent_node = volume.parent.parent
        skip_first_p = False # the first element is not a paper, more like an official summary or introduction
        logger.info(f"Processing volume: {volume_title}")
        if not start and 'Proceedings of the 5th Workshop on Computational Approaches to Historical Language Change' not in volume_title:
            continue
        start = True
        for child in tqdm.tqdm(parent_node.find_all(recursive=False)):
            if child.name == 'p':
                if skip_first_p:
                    meta = parse_acl_paper_meta(child)
                    next_sibling = child.find_next_sibling()
                    abstract = parse_acl_paper_abstract(next_sibling)
                    if abstract is None:
                        if meta['title'] in errata:
                            abstract = errata[meta['title']]['abstract']
                        else:
                            logger.warning(f"Abstract not found for paper `{meta['title']}`.")
                    meta['abstract'] = abstract
                    if meta['uuid'] in uuid2papers:
                        logger.warning(f"UUID {meta['uuid']} already exists in the UUID -> paper mappings.")
                    uuid2papers[meta['uuid']] = meta
                    download_and_rename_pdf(meta, paper_dir)
                else: skip_first_p = True

        # save the UUID -> paper mappings after finishing each volume
        with open(uuid2papers_path, 'w', encoding='utf8') as ouf:
            json.dump(uuid2papers, ouf, ensure_ascii=False, indent=4)
    return uuid2papers


if __name__ == '__main__':

    for url in ['https://aclanthology.org/events/acl-2024/']:
        crawl_acl_anthology_papers(url, output_dir=AIRQA_DIR)
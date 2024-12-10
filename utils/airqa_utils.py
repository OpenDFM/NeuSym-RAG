""" This module contains utility functions for the AIR-QA dataset.
"""
import requests, shutil, subprocess
import tqdm, fitz
import os, sys, re, json, logging
from typing import Dict, Any, Optional, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.models import get_llm_single_instance
from utils.functions.ai_research_metadata import AIRQA_DIR, get_airqa_paper_uuid, get_airqa_paper_metadata, get_num_pages, download_paper_pdf, get_airqa_relative_path, add_ai_research_metadata, write_ai_research_metadata_to_json
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


abbrev_mappings = {
    "conll": "CoNLL",
    "semeval": "SemEval",
    "neurips": "NeurIPS",
    "corl": "CoRL",
    "recsys": "RecSys",
    "automl": "AutoML",
    "collas": "CoLLAs"
}


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


def generate_airqa_example_template() -> Dict[str, Any]:
    """ Generate an AIR-QA example template.
    """
    flag, existing_uids = True, [os.path.splitext(f)[0] for f in os.listdir(os.path.join(AIRQA_DIR, 'examples')) if f.endswith('.json')]
    while flag:
        uid = get_uuid(name=os.path.abspath(__file__) + str(os.urandom(8)))
        if uid not in existing_uids:
            break
    example_template = {
        "uuid": uid,
        "question": "",
        "answer_format": "Your answer should be ",
        "tags": [],
        "pdf_id": [],
        "conference": [],
        "reasoning_steps": [],
        "evaluator": {
            "eval_func": "eval_",
            "eval_kwargs": {}
        },
        "state": {
            "gui-gpt-4o-2024-11-20": False
        }
    }
    with open(os.path.join(AIRQA_DIR, 'examples', uid + '.json'), 'w', encoding='utf8') as ouf:
        json.dump(example_template, ouf, ensure_ascii=False, indent=4)
    logger.info(f"Generated an AIR-QA example template with ID {uid} into examples/{uid}.json file.")
    return example_template


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
    client = get_llm_single_instance(model_name=model)
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


def make_airqa_dataset(airqa_dir: str = AIRQA_DIR, airqa_100: bool = True):
    """ Given all examples UUID json files, merge them into a single JSONL file.
    """
    output_path = os.path.join(airqa_dir, 'test_data.jsonl')
    indir = os.path.join(airqa_dir, 'examples')
    json_files = os.listdir(indir)
    count = 0
    with open(output_path, 'w', encoding='utf8') as of:
        for fp in json_files:
            fp = os.path.join(indir, fp)
            with open(fp, 'r', encoding='utf8') as inf:
                data = json.load(inf)
            if airqa_100 and (len(data['pdf_id']) > 1 or data['uuid'] in [
                'e87fa3e0-7d2f-5909-8e01-5c2d8de2e64c', '398ee3a7-26c8-5967-8b5b-196b5d7641b3', '5c49a736-420a-52b4-8188-ad80f375e948', 'a2985096-8453-5fb7-9066-6f505c734248', 'e1180112-dc52-5a5c-9907-6d007f17b729', '76dc78aa-daa0-5e3a-8377-96072b98e408',
                '8cc38e05-20e5-5a69-8b82-ecc09c03450a', '82bdaa47-a2cb-5fbd-a827-83d981f4bb52',
                '44db1f84-1791-509e-91ae-79b2856153ee', '18b13577-3570-5e5f-be1c-77606cce3cf4',
                "6bb32702-f9f0-53a5-a534-be38bfc75b3f"
            ]): continue
            of.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1
    logger.info(f"Merge {count} AIR-QA examples into {output_path}.")
    return output_path


def make_airqa_metadata(airqa_dir: str = AIRQA_DIR):
    """ Given all metadata JSON files, merge them into a single JSON file.
    """
    output_path = os.path.join(airqa_dir, 'uuid2papers.json')
    uuid2papers = get_airqa_paper_metadata()
    with open(output_path, 'w', encoding='utf8') as ouf:
        json.dump(uuid2papers, ouf, ensure_ascii=False, indent=4)
    logger.info(f"Merge {len(uuid2papers)} AIR-QA metadata into {output_path}.")
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


def crawl_acl_anthology_papers(
        url: str = "https://aclanthology.org/events/acl-2024/",
        model: str = 'gpt-4o',
        temperature: float = 0.0,
        tldr_max_length: int = 80,
        tag_number: int = 5,
        errata_file: Optional[str] = os.path.join(AIRQA_DIR, 'errata.json')
):
    """ Crawl papers from ACL Anthology website. Use `get_airqa_paper_uuid` to generate UUIDs and rename the output PDFs under `output_dir`. Save the meta data dict with its UUID into metadata folder.
    @param:
        url: str, the URL of the ACL Anthology. This parameter will be used to download the HTML file into `html_path`.
        output_dir: str, the directory to save the PDF files.
        errata_file: str, the file path to the errata JSON file.
    """
    # get UUID -> paper mappings
    uuid2papers = get_airqa_paper_metadata()

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
    conference_full = soup.select_one('h2#title').get_text().strip() # e.g., Annual Meeting of the Association for Computational Linguistics
    conference_full = re.sub(r"\s+\(\d+\)", "", conference_full).strip() # remove the year in the conference title
    conference, year = os.path.basename(url.rstrip('#').rstrip(os.sep)).lower().split('-')
    paper_dir = os.path.join(AIRQA_DIR, 'papers', conference + year) # folder to save all paper PDFs


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

                meta['pdf_path'] = get_airqa_relative_path(download_path)
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
                    add_ai_research_metadata(meta, model=model, temperature=temperature, tldr_max_length=tldr_max_length, tag_number=tag_number)
                    write_ai_research_metadata_to_json(meta)
                    uuid2papers[meta['uuid']] = meta
                else: skip_first_p = True

    return uuid2papers


if __name__ == '__main__':

    for url in ['https://aclanthology.org/events/acl-2023/', 'https://aclanthology.org/events/acl-2024/']:
        crawl_acl_anthology_papers(url)
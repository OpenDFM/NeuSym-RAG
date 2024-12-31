""" This module contains utility functions for the AIR-QA dataset.
"""
import requests, shutil, subprocess
import tqdm, fitz
import multiprocessing as mp
from pathlib import Path
import os, sys, re, json, logging
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from typing import Dict, Any, Optional, List, Tuple, Union
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.models import get_llm_single_instance
from utils.functions.ai_research_metadata import AIRQA_DIR, get_airqa_paper_uuid, get_airqa_paper_metadata, get_num_pages, download_paper_pdf, get_airqa_relative_path, add_ai_research_metadata, write_ai_research_metadata_to_json, infer_paper_abstract_from_pdf, infer_paper_authors_from_pdf
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
        output_file: str = os.path.join(AIRQA_DIR, 'used_paper_uuids.json')
    ) -> List[str]:
    """ Extract all used paper UUIDs from the AIR-QA examples.
    """
    uuids = set()
    for file in os.listdir(example_dir):
        if file.endswith('.json'):
            with open(os.path.join(example_dir, file), 'r', encoding='utf8') as inf:
                data = json.load(inf)
                uuids.update(data['anchor_pdf'])
                uuids.update(data['reference_pdf'])
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
        "anchor_pdf": [],
        "reference_pdf": [],
        "conference": [],
        "reasoning_steps": [],
        "evaluator": {
            "eval_func": "eval_",
            "eval_kwargs": {}
        },
        "state": {
            "gui-gpt-4o-2024-11-20": False
        },
        "annotator": "human"
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


def make_airqa_dataset(airqa_dir: str = AIRQA_DIR, airqa_100: bool = False) -> str:
    """ Given all examples UUID json files, merge them into a single JSONL file.
    """
    uuids_to_include = []
    if airqa_100:
        output_path = os.path.join(airqa_dir, 'test_data_airqa100.jsonl')
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf8') as inf:
                for line in inf:
                    data = json.loads(line)
                    uuids_to_include.append(data['uuid'])
    else:
        output_path = os.path.join(airqa_dir, 'test_data.jsonl')

    indir = os.path.join(airqa_dir, 'examples')
    json_files = os.listdir(indir)
    count = 0
    with open(output_path, 'w', encoding='utf8') as of:
        for fp in json_files:
            fp = os.path.join(indir, fp)
            with open(fp, 'r', encoding='utf8') as inf:
                data = json.load(inf)
            if airqa_100 and data['uuid'] not in uuids_to_include: continue
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
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0,
        tldr_max_length: int = 80,
        tag_number: int = 5
) -> Dict[str, dict]:
    """ Crawl papers from ACL Anthology website. Use `get_airqa_paper_uuid` to generate UUIDs and rename the output PDFs under `output_dir`. Save the meta data dict with its UUID into metadata folder.
    @param:
        url: str, the URL of the ACL Anthology. This parameter will be used to download the HTML file into `html_path`.
        output_dir: str, the directory to save the PDF files.
        errata_file: str, the file path to the errata JSON file.
    """
    # get UUID -> paper mappings
    uuid2papers = get_airqa_paper_metadata()

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
    logger.info(f"Processing conference ({conference}-{year}): {conference_full}")

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
                if meta['uuid'] in uuid2papers:
                    return uuid2papers[meta['uuid']]
                download_path = os.path.join(paper_dir, meta['uuid'] + '.pdf')
                if not download_paper_pdf(meta['pdf_url'], download_path, log=False):
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
                        # logger.warning(f"UUID {meta['uuid']} already exists in the UUID -> paper mappings {meta['title']}.")
                        continue
                    add_ai_research_metadata(meta, model=model, temperature=temperature, tldr_max_length=tldr_max_length, tag_number=tag_number)
                    write_ai_research_metadata_to_json(meta)
                    uuid2papers[meta['uuid']] = meta
                else: skip_first_p = True

    return uuid2papers


def crawl_openreview_papers(
        conference: str,
        year: int,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0,
        tldr_max_length: int = 80,
        tag_number: int = 5,
        num_processes: Optional[int] = None
    ) -> Dict[str, dict]:
    """ Refer to blogs and doc:
        - https://blog.csdn.net/qq_39517117/article/details/142959952
        - https://github.com/fedebotu/ICLR2023-OpenReviewData
        - https://docs.openreview.net/getting-started/using-the-api
    Please firstly: pip install openreview-py
    And set the environment variable OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD.
    """
    try:
        import openreview
        from openreview.api import OpenReviewClient, Note
    except ImportError:
        logger.error("Please pip install the openreview-py package firstly.")
        return {}
    uuid2papers = get_airqa_paper_metadata()
    if os.environ.get('OPENREVIEW_USERNAME', None) is None or os.environ.get('OPENREVIEW_PASSWORD', None) is None:
        logger.error("Please set the environment variable OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD.")
        return uuid2papers

    def get_openreview_client_and_venues(conference: str, year: int) -> Tuple[Union[openreview.Client, OpenReviewClient], List[str]]:
        conference = conference.lower()
        if conference == 'iclr' and int(year) <= 2023: # api2 failed to retrieve content in ICLR 2023
            client: openreview.Client = openreview.Client(baseurl='https://api.openreview.net')
        else:
            client: OpenReviewClient = OpenReviewClient(baseurl='https://api2.openreview.net')
        venues = client.get_group(id='venues').members
        venues = [v for v in venues if v.lower().startswith(f'{conference}.cc/{year}/') and 'blog' not in v.lower()]
        return client, venues

    client, venues = get_openreview_client_and_venues(conference, year)
    conference_full = 'International Conference on Learning Representations' if conference.lower() == 'iclr' \
        else 'Conference on Neural Information Processing Systems' if conference.lower() in ['nips', 'neurips'] \
        else conference
    subfolder = os.path.join(AIRQA_DIR, 'papers', conference.lower() + str(year))
    logger.info(f"In total, {len(venues)} venues found for {conference} {year}.")
    processed_data = []
    for venue_id in venues:
        logger.info(f"Processing volume: {venue_id} ...")
        output_venue_path = f"{venue_id.replace('/', '_').replace('-', '_').lower()}.json"
        if os.path.exists(output_venue_path):
            with open(output_venue_path, 'r', encoding='utf8') as inf:
                submissions = json.load(inf)
        else:
            submissions: List[Note] = client.get_all_notes(content={'venueid': venue_id})
            submissions = [submit.to_json() for submit in submissions]
            # we only keep accepted papers with downloadable PDFs
            filtered = []
            for data in submissions:
                # type(data['content']['venue']) == dict -> {'value': 'xxx'}, o.w., data['content']['venue'] == str
                if type(data['content']['venue']) == dict and 'submitted to' not in data['content']['venue']['value'].lower() and data['content'].get('pdf', {}).get('value', None):
                    for k in data['content']:
                        if type(data['content'][k]) == dict and 'value' in data['content'][k]:
                            data['content'][k] = data['content'][k]['value']
                    filtered.append(data)
                elif type(data['content']['venue']) != dict and 'submitted to' not in data['content']['venue'].lower() and data['content'].get('pdf', None):
                    filtered.append(data)
            submissions = filtered 
            # submissions = [data for data in submissions if 'submitted to' not in data['content']['venue'].lower() and data['content'].get('pdf', None)]
            if len(submissions) == 0:
                logger.warning(f"No papers found for {venue_id}.")
                continue
            # serialize first to prevent follow-up processing errors
            with open(output_venue_path, 'w', encoding='utf8') as ouf:
                json.dump(submissions, ouf, ensure_ascii=False, indent=4)

        logger.info(f"In total, {len(submissions)} submissions found for venue {venue_id}.")
        for submission in tqdm.tqdm(submissions):
            data = submission['content']
            title = str(data['title']).strip()
            uid = get_airqa_paper_uuid(title, f'{conference.lower()}{year}')

            if uid in uuid2papers:
                logger.warning(f"UUID {uid} already exists in the UUID -> paper mappings: {title}")
                continue

            pdf_url = data['pdf'] if data['pdf'].startswith('http') else 'https://openreview.net/' + data['pdf'].lstrip('/')
            metadata = {
                "uuid": uid, # UUID generated by the title
                "title": title, # paper title
                "conference_full": conference_full, # full title of the conference
                "conference": conference,
                "year": int(year), # conference year
                "volume": data['venue'], # volume title
                "bibtex": data['_bibtex'], # bibtex citation
                "authors": data.get('authors', None), # authors list
                "pdf_url": pdf_url, # URL to download the PDF
                "pdf_path": os.path.join(subfolder, f'{uid}.pdf'), # local path to save the PDF, rename it with the UUID
                "num_pages": -1, # number of pages in the PDF
                "abstract": data.get('abstract', None), # paper abstract
                "tldr": data.get('TLDR', None), # TLDR
                "tags": data.get('keywords', None) # tags
            }
            processed_data.append(metadata)
            uuid2papers[metadata['uuid']] = metadata

    # download all papers from pdf_url to pdf_path using multi-processing
    download_openreview_papers(processed_data, subfolder, num_processes=num_processes)

    # post-process the metadata: num_pages, relative_path, abstract, TLDR, tags
    logger.info(f"Post-processing the metadata for {len(processed_data)} papers ...")
    for metadata in tqdm.tqdm(processed_data):
        try:
            metadata['num_pages'] = get_num_pages(metadata['pdf_path'])
            if not metadata['abstract']:
                metadata['abstract'] = infer_paper_abstract_from_pdf(metadata['pdf_path'], model=model, temperature=temperature)
            if not metadata['authors']:
                metadata['authors'] = infer_paper_authors_from_pdf(metadata['pdf_path'], model=model, temperature=temperature)
            add_ai_research_metadata(metadata, model=model, temperature=temperature, tldr_max_length=tldr_max_length, tag_number=tag_number)
            metadata['pdf_path'] = get_airqa_relative_path(metadata['pdf_path'])
            write_ai_research_metadata_to_json(metadata)
        except Exception as e:
            logger.error(f"Failed to post-process the metadata for {metadata['uuid']}: {e}")
    return uuid2papers


def handle_download_retry_error(retry_state):
    logger.info(f"All retries failed. Last exception: {retry_state.outcome.exception()}")
    return None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), retry_error_callback=handle_download_retry_error)
def multiprocessing_download_wrapper(args):
    pdf_url, pdf_path = args
    pdf_path = download_paper_pdf(pdf_url, pdf_path, log=False)
    if pdf_path is None:
        raise ValueError(f"Failed to download the PDF file from {pdf_url} into {pdf_path}.")
    return pdf_path


def download_openreview_papers(processed_data: List[Dict[str, Any]], parent_folder: str, num_processes: Optional[int] = None) -> int:
    """ Download all papers from OpenReview.
    """
    count = sum(1 for _ in Path(parent_folder).rglob('*') if _.is_file())
    # download all papers in parallel
    papers_to_download = [(metadata['pdf_url'], metadata['pdf_path']) for metadata in processed_data]
    num_processes = int(0.6 * mp.cpu_count()) if num_processes is None else num_processes

    logger.info(f"Starting to download {len(papers_to_download)} papers from OpenReview ({count} already exist) ...")
    with mp.Pool(num_processes) as pool:
        pool.map(multiprocessing_download_wrapper, papers_to_download)
    count = sum(1 for _ in Path(parent_folder).rglob('*') if _.is_file()) - count
    logger.info(f"Finished downloading {len(papers_to_download)} papers from OpenReview ({count} new downloaded) .")
    return count


if __name__ == '__main__':

    from itertools import combinations

    # for url in ['https://aclanthology.org/events/acl-2023/', 'https://aclanthology.org/events/acl-2024/', 'https://aclanthology.org/events/emnlp-2023/', 'https://aclanthology.org/events/emnlp-2024/']:
    #     crawl_acl_anthology_papers(url, model='gpt-4o-mini', temperature=0.0, tldr_max_length=80, tag_number=5)

    # for conference, year in combinations(['ICLR', 'NeurIPS'], [2023, 2024]):
    #     crawl_openreview_papers(conference, year, model='gpt-4o-mini', temperature=0.0, tldr_max_length=80, tag_number=5)
    crawl_openreview_papers('ICLR', 2024, model='gpt-4o-mini', temperature=0.0, tldr_max_length=80, tag_number=5)
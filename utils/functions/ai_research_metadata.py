#coding=utf8
import html, json, requests, shutil, uuid, tempfile, subprocess, sys, os, re, logging, time
import urllib, urllib.request
import xmltodict
from bs4 import BeautifulSoup
from typing import List, Union, Optional, Tuple, Any, Dict
import pymupdf
from fuzzywuzzy import fuzz
import pandas as pd
from urllib.parse import urlencode, quote

from utils.functions.common_functions import is_valid_uuid, get_uuid, call_llm, call_llm_with_message, convert_to_message
from utils.functions.parallel_functions import parallel_extract_or_fill


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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'dataset', 'airqa'
)

TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tmp'
)


UUID2PAPERS: Dict[str, Any] = {}
CCF_CONFERENCES: Optional[Dict[str, Any]] = None


def get_airqa_paper_uuid(title: str, conference_year: str = 'uncategorized') -> str:
    """ Get the UUID of a paper in the AIR-QA dataset.
    - `title` should be normalized before getting UUID;
    - `meta` is lowercase {conference}{year}, e.g., 'acl2024', or special 'unauthorized'.
    """
    # normalize the paper title
    conference_year = conference_year.lower()
    assert re.match(r'^[a-z\d]+$', conference_year), f"Invalid conference_year: `{conference_year}`, can only contain letters and digits."
    title = re.sub(r'[^a-z\d]', '', title.lower())
    paper = title + '-' + conference_year.lower()
    return get_uuid(paper, uuid_type='uuid5', uuid_namespace='dns')


def get_airqa_paper_metadata(uuid_str: Optional[str] = None, dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    """ Get the metadata dict of a paper in the AIR-QA dataset.
    """
    if dataset_dir is not None:
        if not os.path.exists(dataset_dir):
            logger.error(f"[Error]: The dataset directory {dataset_dir} does not exist.")
            return None
    else: dataset_dir = AIRQA_DIR
    global UUID2PAPERS
    if not UUID2PAPERS.get(dataset_dir):
        UUID2PAPERS[dataset_dir] = {}
        metadata_dir = os.path.join(dataset_dir, 'metadata')
        files = os.listdir(metadata_dir)
        for f in files:
            fp = os.path.join(metadata_dir, f)
            if fp.endswith('.json') and is_valid_uuid(os.path.basename(fp).split('.')[0]):
                with open(fp, 'r', encoding='utf-8') as inf:
                    paper_dict = json.load(inf)
                    UUID2PAPERS[dataset_dir][paper_dict['uuid']] = paper_dict
    if uuid_str is not None:
        assert is_valid_uuid(uuid_str) and uuid_str in UUID2PAPERS[dataset_dir], f"Invalid UUID string: {uuid_str}."
        return UUID2PAPERS[dataset_dir][uuid_str]
    return UUID2PAPERS[dataset_dir]


def download_paper_pdf(pdf_url: str, pdf_path: str, log: bool = True) -> Optional[str]:
    """ Download the PDF file from the `pdf_url` into `pdf_path`. Just return the relative `pdf_path` if succeeded.
    """
    if os.path.exists(pdf_path) and os.path.isfile(pdf_path): # PDF file already exists
        if log: logger.warning(f"PDF file {pdf_path} already exists. Just ignore the download from {pdf_url}.")
        return pdf_path
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        } # simulate user agent
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        response = requests.get(pdf_url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            if log: logger.info(f"Downloaded paper `{pdf_url}` successfully to: {pdf_path}")
            return pdf_path
        else:
            if log: logger.error(f"Failed to download paper `{pdf_url}` to `{pdf_path}`. Status code: {response.status_code}")
            else: print(f"Failed to download paper `{pdf_url}` to `{pdf_path}`. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        if log: logger.error(f"An error occurred while downloading PDF file: {e}")
        else: print(f"An error occurred while downloading PDF file: {e}")
    return None


def get_airqa_relative_path(file_path: str) -> str:
    """ Get the relative path of a file.
    """
    working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    doc = pymupdf.open(pdf_path)
    num_pages = doc.page_count
    doc.close()
    if num_pages == 0:
        repair_pdf_with_qpdf(pdf_path)
        doc = pymupdf.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    return num_pages


def get_ccf_conference_name(conference_abbrev: str = None, conference_full: str = None) -> str:
    """ Load the CCF conference dataframe from the provided CSV file.
    """
    def normalize_conference_name(cf_name: str) -> str:
        return re.sub(r'[^a-z\d]', '', cf_name.lower().replace('&', 'and'))

    global CCF_CONFERENCES
    if CCF_CONFERENCES is None:
        ccf_file: str = os.path.join(AIRQA_DIR, 'ccf_catalog.csv')
        ccf_pd = pd.read_csv(ccf_file)

        CCF_CONFERENCES = {
            "abbrev2full": {
                str(row['abbr']).lower(): str(row['name']) for idx, row in ccf_pd.iterrows() if not pd.isna(row['abbr']) and not pd.isna(row['name'])
            },
            "full2abbrev": {
                normalize_conference_name(str(row['name'])): str(row['abbr']) for idx, row in ccf_pd.iterrows() if not pd.isna(row['abbr']) and not pd.isna(row['name'])
            }
        }
    
    if conference_abbrev is not None:
        return conference_abbrev, CCF_CONFERENCES["abbrev2full"].get(str(conference_abbrev).lower(), conference_abbrev)
    else:
        return CCF_CONFERENCES["full2abbrev"].get(normalize_conference_name(conference_full), conference_full), conference_full


def infer_paper_title_from_pdf(
        pdf_path: str,
        first_lines: Optional[int] = None,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0 # Use more deterministic decoding with temperature=0.0
    ) -> str:
    """ Use a language model to infer the title of a paper from the top `first_lines` lines of the first page in a PDF.
    """
    doc = pymupdf.open(pdf_path)
    first_page = doc[0]
    if first_lines is not None:
        first_page_text = '\n'.join(first_page.get_text().split('\n')[:first_lines])
        first_lines = f"the top {first_lines} lines of the first page"
    else:
        first_page_text = first_page.get_text()
        first_lines = "the first page"
    doc.close()

    # Call the language model to infer the title
    template = f"""You are an expert in academic papers. Your task is to identify the raw title of a research paper based on the extracted text from the first page. The provided text is from {first_lines} in the PDF file, extracted using PyMuPDF. Please ensure the following:\n1. Directly return the title without adding any extra context, explanations, or formatting.\n2. Do not modify the raw title—retain its original capitalization and punctuation exactly as presented.\n3. If the title spans multiple lines, concatenate them into a single line and return it.\n4. If you are certain that the provided text does not contain the paper's title, respond only with "title not found".\n\nHere is the extracted text:\n\n{first_page_text}\n\nYour response is:
    """
    title = call_llm(template, model=model, temperature=temperature).strip()
    if title.startswith("title not found"):
        logger.error(f"Paper title is not found in {first_lines} of the PDF {pdf_path}.")
        return None
    return title


def infer_paper_volume_from_pdf(
        pdf_path: str,
        first_lines: Optional[int] = 10,
        last_lines: Optional[int] = 10,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0, # Use more deterministic decoding with temperature=0.0
        **kwargs
    ) -> str:
    """ Use a language model to infer the volume of a paper from the top `first_lines` lines and the bottom `last_lines` of the first page in a PDF.
    """
    if kwargs.get("ignore_llm"): return None
    volume_prompt_path = os.path.join(os.path.dirname(__file__), "volume_prompt.json")
    with open(volume_prompt_path, "r", encoding='utf-8') as f:
       VOLUME_PROMPTS = json.load(f)
    VOLUME_SYSTEM_PROMPT = VOLUME_PROMPTS["VOLUME_SYSTEM_PROMPT"]
    VOLUME_USER_PROMPT = VOLUME_PROMPTS["VOLUME_USER_PROMPT"]
    VOLUME_FEW_SHOTS = VOLUME_PROMPTS["VOLUME_FEW_SHOTS"]
    doc = pymupdf.open(pdf_path)
    first_page = doc[0]
    first_lines_text = '\n'.join(first_page.get_text().split('\n')[:first_lines])
    last_lines_text = '\n'.join(first_page.get_text().split('\n')[-last_lines:])
    doc.close()

    # Call the language model to infer the title
    messages = [{
        "role": "system",
        "content" : VOLUME_SYSTEM_PROMPT
    }]
    for example in VOLUME_FEW_SHOTS:
        messages.append({
            "role": "user", 
            "content": VOLUME_USER_PROMPT.format(
                first_lines_text=example['first_lines_text'],
                last_lines_text=example['last_lines_text']
            )
        })
        messages.append({
            "role": "assistant",
            "content": example['volume']
        })
    messages.append({
        "role": "user",
        "content": VOLUME_USER_PROMPT.format(first_lines_text=first_lines_text, last_lines_text=last_lines_text)
    })
    volume = call_llm_with_message(messages=messages, model=model, temperature=temperature).strip()
    if volume.lower().startswith("volume not found"):
        logger.error(f"Paper volume is not found in the first {first_lines} and last {last_lines} lines of PDF {pdf_path}.")
        return None
    return volume


def infer_paper_abstract_from_pdf(
        pdf_path: str,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0, # Use more deterministic decoding with temperature=0.0
        **kwargs
    ) -> str:
    """ Use a language model to infer the abstract of a paper from the first page in a PDF.
    """
    if kwargs.get("ignore_llm"): return None
    doc = pymupdf.open(pdf_path)

    template = """You are an expert in academic papers. Your task is to identify the abstract of a research paper based on the text from the first {page_num} page(s) in the PDF, extracted using PyMuPDF. Please ensure the following:\n1. Directly return the abstract without adding any extra context, explanations, or formatting.\n2. Do not modify the abstract, retain its original capitalization and punctuation exactly as presented.\n3. If the abstract spans multiple lines, concatenate them into a single line and return it.\n4. If you are certain that the provided text does not contain the paper's abstract, respond only with "abstract not found".\n\nHere is the extracted text:\n\n```txt\n{page_content}\n```\n\nYour response is:\n"""
    
    page_content = ""
    for page_num in range(0, min(4, len(doc))):
        page_content += doc[page_num].get_text()
        abstract = call_llm(template.format(page_num=page_num+1, page_content=page_content), model=model, temperature=temperature).strip()
        if not abstract.startswith("abstract not found"):
            doc.close()
            return abstract
    logger.error(f"Paper abstract is not found in the first page of the PDF {pdf_path}.")
    doc.close()
    return None


def infer_paper_authors_from_pdf(
        pdf_path: str,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0 # Use more deterministic decoding with temperature=0.0
    ) -> List[str]:
    """ Use a language model to infer the authors of a paper from the first page in a PDF.
    """
    doc = pymupdf.open(pdf_path)
    first_page = doc[0]
    first_page_text = first_page.get_text()
    doc.close()

    # Call the language model to infer the authors
    template = f"""You are an expert in academic papers. Your task is to identify the authors of a research paper based on the extracted text from the first page in the PDF file, using PyMuPDF. Please ensure the following:\n1. Directly return the author list (of Python type `List[str]`) without adding any extra context, explanations, or formatting, e.g., ["abc", "xyz"]\n2. Do not modify the author names, which means preserving their original capitalization and order exactly as presented (from left to right, from top to bottom).\n3. If you are certain that the provided text does not contain the paper's authors, respond only with "None".\n\nHere is the extracted text:\n\n{first_page_text}\n\nYour response is:
    """
    authors = call_llm(template, model=model, temperature=temperature).strip()
    if authors.lower().strip() == "none":
        logger.error(f"Paper authors are not found in the first page of the PDF {pdf_path}.")
        return []
    try:
        authors = eval(authors.strip())
        if type(authors) == list:
            return authors
        else:
            raise ValueError()
    except Exception as e:
        logger.error(f"Failed to parse valid author list from {authors} .")
        return []


def infer_paper_tldr_from_metadata(
        pdf_title: str,
        pdf_abstract: str,
        max_length: int = 60,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
    """ Use a language model to infer the TL;DR of a paper based on its title and abstract.
    """
    # Call the language model to infer the TL;DR
    template = f"""You are an expert in academic papers. Your task is to write a TL;DR (Too Long; Didn't Read) summary for a research paper based on its title and abstract. The TL;DR should:\n1. Be concise and within {max_length} characters.\n2. Capture the main focus or contribution of the paper.\n3. Be written in a single line without extra formatting or context.\n\nHere are the title and abstract of the paper.\nTitle: {pdf_title}\nAbstract: {pdf_abstract}\n\nYour response is:"""
    if kwargs.get("parallel"):
        tldr = parallel_extract_or_fill(template, **kwargs).strip()
    else:
        tldr = call_llm(template, model=model, temperature=temperature, top_p=top_p).strip()
    return tldr


def infer_paper_tags_from_metadata(
        pdf_title: str,
        pdf_abstract: str,
        tag_number: int = 5,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
    """ Use a language model to infer tags (keywords) of a paper based on its title and abstract.
    """
    # Call the language model to infer the tags
    template = f"""You are an expert in academic papers. Your task is to generate a list of {tag_number} relevant tags (keywords) for a research paper based on its title and abstract. The tags should:\n1. Be concise and relevant to the paper's main focus.\n2. Be unique (avoid duplicates).\n3. Be written as a comma-separated list.\n\nHere are the title and abstract of the paper.\nTitle: {pdf_title}\nAbstract: {pdf_abstract}\nYour response is:\n"""
    tags = None
    if kwargs.get("parallel"):
        tags = parallel_extract_or_fill(template, **kwargs).strip()
    else:
        tags = call_llm(template, model=model, temperature=temperature, top_p=top_p, **kwargs).strip()
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
    return tag_list


def extract_metadata_from_scholar_api(
        title: str,
        api_tools: List[str] = ['arxiv', 'dblp', 'semantic-scholar'],
        **kwargs
    ) -> Dict[str, Any]:
    """ Given the title or the arxiv id of one paper, extract its metadata from provided scholar APIs.
    @param:
        title: str, the title of the paper
        api_tool: List[str], the list of scholar API tools to use, each element is chosen from
            ['dblp', 'semantic-scholar', 'arxiv']
    """
    for tool in api_tools:
        assert tool in ['arxiv', 'dblp', 'semantic-scholar'], f"Invalid scholar API tool: {tool}."
    if not api_tools: # try sequentially with pre-defined orders
        api_tools = ['arxiv', 'dblp', 'semantic-scholar']
    functions = {
        "arxiv": arxiv_scholar_api,
        "dblp": dblp_scholar_api,
        "semantic-scholar": semantic_scholar_api,
    }
    # Call the scholar API to extract the metadata
    metadata_dict = {}
    for tool in api_tools:
        metadata_dict = functions[tool](title, **kwargs)
        if metadata_dict is not None:
            return metadata_dict
    logger.error(f'[Error]: failed to extract the metadata information for paper `{title}` from these provided scholar APIs: {api_tools}.')
    return None


def arxiv_scholar_api(arxiv_id_or_title: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """ Given the arxiv_id of one paper, extract its metadata from arxiv API.
    @param:
        arxiv_id_or_title: str, the arxiv_id or title of the paper
        **kwargs: Dict[str, Any], other arguments that will be directly passed to the arxiv API
            - limit: int, the maximum number of search results to return, by default 10
            - threshold: int, the threshold of the fuzzy ratio to filter the search results, by default 90
            - dataset_dir: str, folder path to the dataset, by default AIRQA_DIR
    @return: metadata dict
        see doc in `get_ai_research_metadata`
    """
    ARXIV_API_URL = 'http://export.arxiv.org/api/query'
    arxiv_id_or_title = arxiv_id_or_title.strip()
    if re.search(r'^\d+\.\d+$', arxiv_id_or_title) or re.search(r'^\d+\.\d+v\d+$', arxiv_id_or_title):
        # arxiv id
        is_arxiv_id = True
        search_query = arxiv_id_or_title
    else:
        is_arxiv_id = False
        normed_title = re.sub(r'[^a-zA-Z0-9]', ' ', arxiv_id_or_title).strip()
        search_query = f"ti:{quote(normed_title)}"
    options = {
        "search_query": search_query,
        "start": 0,
        "max_results": max(1, kwargs.get('limit', 10))
    }
    try:
        xml_response = requests.get(ARXIV_API_URL, params=options)
        xml_response = xml_response.content.decode('utf8')
    except Exception as e:
        logger.error(f"An unexpected error occurred during calling ARXIV API to search `{arxiv_id_or_title}`: {e}")
        return None
    
    data = xmltodict.parse(xml_response).get("feed", {}).get("entry", {})
    if data == {} or data == []:
        logger.error(f"Failed to find paper {arxiv_id_or_title} using ARXIV API.")
        return None

    def select_return_result(data: List[dict]) -> dict:
        if type(data) == dict: data = [data]
        normed_id_or_title = arxiv_id_or_title.lower().strip()
        filtered_data = []
        if is_arxiv_id:
            normed_id_or_title = re.sub(r'v\d+$', '', normed_id_or_title)
        for hit in data:
            if is_arxiv_id: # require exact match for arxiv id
                extracted_id = hit["id"].split('/')[-1]
                if extracted_id.startswith(normed_id_or_title):
                    hit['fuzzy-score'] = fuzz.ratio(extracted_id, normed_id_or_title)
                    filtered_data.append(hit)
            else: # allow fuzzy matching for title with threshold
                hit_title = html.unescape(hit.get('title', '').lower().replace('\n ', '').strip())
                hit['fuzzy-score'] = fuzz.ratio(normed_id_or_title, hit_title)
                if hit['fuzzy-score'] >= kwargs.get('threshold', 90):
                    filtered_data.append(hit)
        return filtered_data

    filtered_data = select_return_result(data)
    if len(filtered_data) == 0:
        logger.warning(f"Not found paper with arxiv id or title `{arxiv_id_or_title}` in ARXIV.")
        return None

    # only the highest score data
    data = sorted(filtered_data, key=lambda x: x['fuzzy-score'], reverse=True)[0]
    year = data["updated"][:4]
    arxiv_id = re.sub(r'v\d+$', '', data['id'].split('/')[-1])
    title, subfolder = re.sub(r'\n ', ' ', re.sub(r'\n  ', ' ', data["title"].strip())), f"arxiv{year}"
    paper_uuid = get_airqa_paper_uuid(title, subfolder)
    if kwargs.get('dataset_dir', None) is not None:
        assert os.path.exists(kwargs['dataset_dir']), f"Invalid dataset directory: {kwargs['dataset_dir']}."
        dataset_dir = kwargs['dataset_dir']
    else: dataset_dir = AIRQA_DIR
    pdf_path = os.path.join(dataset_dir, 'papers', subfolder, f'{paper_uuid}.pdf')
    pdf_url = data['id'].replace('/abs/', '/pdf/') # do not use arxiv id, directly use the pdf link in the returned id
    authors = [data["author"]["name"]] if type(data["author"]) == dict and 'name' in data["author"] else [author["name"] for author in data["author"]]
    abstract = data["summary"]

    def get_arxiv_bibtex(data):
        # construct bibtex automatically
        ref = authors[0].strip().split(' ')[-1] + year + ''.join(title.split(' ')[:5])
        bibtex = '@misc{' + re.sub(r'[^a-z0-9]', '', ref.lower()) + ',\n'
        bibtex += f'    title = {{{title}}},\n'
        bibtex += f'    author = {{{" and ".join(authors)}}},\n'
        bibtex += f'    year = {{{year}}},\n'
        bibtex += f'    eprint = {{{arxiv_id}}},\n'
        bibtex += '    archivePrefix = {arXiv},\n'
        primary_class = data.get("arxiv:primary_category", {}).get('@term', 'cs.AI')
        bibtex += f'    primaryClass = {{{primary_class}}},\n'
        url = '/'.join(data['id'].split('/')[:-1]) + f'/{arxiv_id}'
        bibtex += f'    url = {{{url}}},\n'
        bibtex += '}'
        return bibtex

    metadata = {
        "uuid": paper_uuid,
        "title": title,
        "conference": "arxiv",
        "conference_full": "ArXiv",
        "volume": None, # no volume
        "year": int(year),
        "authors": authors,
        "pdf_url": pdf_url,
        "pdf_path": pdf_path,
        "bibtex": get_arxiv_bibtex(data),
        "abstract": abstract
    }
    return metadata


def semantic_scholar_api(title: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """ Given the title of one paper, extract its metadata from Semantic Scholar API. We resort to the following endpoint:
    Paper bulk search: https://api.semanticscholar.org/graph/v1/paper/search/bulk
        - Paper title search only returns one single result, not suitable since we may request that open access is true
        - According to official tutorial, https://www.semanticscholar.org/product/api/tutorial#step-1-guide, we should mostly use paper bulk search because it is more efficient (it seldom encounters the "Too Many Requests" error). Probaly need Semantic Scholar API key for abundant usage (set environment variable `S2_API_KEY`) if "Paper relevance search" (/paper/search) is expected.
    @param:
        title: str, the title of the paper
        **kwargs: Dict[str, Any], other arguments that will be directly passed to the DBLP API
            - limit: int, the maximum number of search results to return, by default 10
            - threshold: int, the threshold of the fuzzy ratio to filter the search results, by default 90
            - fields_of_study: List[str], the list of fields of study to filter the search results, e.g., ['Computer Science', 'Linguistics']. By default, no filter
            - start_year: int, the start year of the publication, used to narrow down search result. By default, None
            - dataset_dir: str, folder path to the dataset, by default AIRQA_DIR
    @return: metadata dict
        see doc in `get_ai_research_metadata`
    """
    api_key = os.environ.get('S2_API_KEY', None)
    headers = {"x-api-key": api_key} if api_key is not None else None
    url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
    query_params = {
        "query": title,
        "openAccessPdf": True,
        "publicationTypes": "Review,JournalArticle,Conference,Dataset",
        "fields": "title,abstract,publicationVenue,year,openAccessPdf,authors,citationStyles"
    }
    if kwargs.get('fields_of_study', None) is not None:
        query_params['fieldsOfStudy'] = ",".join(kwargs['fields_of_study'])
    if kwargs.get('start_year', None) is not None:
        query_params['year'] = f'{kwargs["start_year"]}-'
    hits = []
    try:
        response = requests.get(url, headers=headers, json=query_params, timeout=60)
        response.raise_for_status()
        response = response.json()
        if response['total'] > 1000:
            logger.warning(f"Too many search results for paper `{title}` in Semantic Scholar API. Better narrow down the search.")
            return None
        data = response['data']
        for data_dict in data:
            hit_title = data_dict['title']
            data_dict['fuzzy-score'] = fuzz.ratio(title.lower(), hit_title.lower())
            if data_dict['fuzzy-score'] >= kwargs.get('threshold', 90):
                hits.append(data_dict)
    except Exception as e:
        logger.error(f"An unexpected error occurred during calling Semantic Scholar API to search `{title}`: {e}")
        return None
    if len(hits) == 0:
        logger.warning(f"Not found paper with title `{title}` and open access PDF in Semantic Scholar.")
        return None

    sorted_hits = sorted(hits, key=lambda x: x['fuzzy-score'], reverse=True)
    for data in sorted_hits:
        try:
            data['title'] = data['title'].strip()
            year = int(data['year'])
            if data.get('publicationVenue', {}):
                conference_full = data['publicationVenue']['name']
                conference, conference_full = get_ccf_conference_name(conference_full=conference_full)
                if conference == conference_full: # indeed not found in the CCF list
                    alternate_names = data['publicationVenue']['alternate_names']
                    conference = sorted(alternate_names, key=lambda x: len(x))[0]
                    if conference == 'NIPS': # special case
                        conference = 'NeurIPS'
                subfolder = conference.lower() + f'{year}' if re.search(r'^[a-z\d]+$', conference.lower()) else 'uncategorized'
            else:
                conference_full, conference, subfolder = None, None, 'uncategorized'
            paper_uid = get_airqa_paper_uuid(data['title'], subfolder)
            assert data['openAccessPdf']['url'] and data['citationStyles']['bibtex'], f"Invalid open access URL or bibtex."
            if kwargs.get('dataset_dir', None) is not None:
                assert os.path.exists(kwargs['dataset_dir']), f"Invalid dataset directory: {kwargs['dataset_dir']}."
                dataset_dir = kwargs['dataset_dir']
            else: dataset_dir = AIRQA_DIR
            metadata_dict = {
                "uuid": paper_uid,
                "title": data['title'],
                "conference": conference,
                "conference_full": conference_full,
                "volume": None,
                "year": year,
                "authors": [author['name'] for author in data['authors']],
                "pdf_url": data['openAccessPdf']['url'],
                "pdf_path": os.path.join(dataset_dir, 'papers', subfolder, f'{paper_uid}.pdf'),
                "bibtex": data['citationStyles']['bibtex'],
                "abstract": data['abstract']
            }
            return metadata_dict
        except Exception as e:
            logger.error(f'Error occurred when trying to process semantic scholar hit: {json.dumps(data)}\n{e}')
            pass
    logger.error(f"Failed to extract metadata for paper with title `{title}` and open access PDF in Semantic Scholar.")
    return None


def dblp_scholar_api(title: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """ Given the title of one paper, extract its metadata from DBLP API.
    @param:
        title: str, the title of the paper
        **kwargs: Dict[str, Any], other arguments that will be directly passed to the DBLP API
            - limit: int, the maximum number of search results to return, by default 10
            - threshold: int, the threshold of the fuzzy ratio to filter the search results, by default 90
            - allow_arxiv: bool, whether to allow arxiv papers in the search results, by default False
            - dataset_dir: str, folder path to the dataset, by default AIRQA_DIR
    @return: metadata dict
        see doc in `get_ai_research_metadata`
    """
    DBLP_BASE_URL = 'https://dblp.org/search/publ/api'
    options = {
        'q': title, # searching method for query: https://dblp.org/faq/1474589.html
        'format': 'json',
        'h': max(1, kwargs.get('limit', 10)) # maximum return result
    } # see parameters: https://dblp.org/faq/How+to+use+the+dblp+search+API.html
    search_url = f'{DBLP_BASE_URL}?{urlencode(options)}'
    try:
        response = requests.get(search_url, timeout=60)
        response.raise_for_status()
        json_response = response.json()
    except Exception as e:
        logger.error(f"An unexpected error occurred during calling DBLP API to search `{search_url}`: {e}")
        return None
    if json_response.get('result', {}).get('hits', {}).get('hit', None) is None:
        logger.warning(f"Not found paper with title `{title}` in DBLP.")
        return None

    # post-processing searched results, success hit only if ratio > threshold
    hits = json_response['result']['hits']['hit']
    threshold = kwargs.get('threshold', 90)
    filtered_hits = []
    for hit in hits: # filter result
        title_hit = html.unescape(hit['info'].get('title', '').rstrip('.'))
        hit['info']['title'] = title_hit
        hit['fuzzy-score'] = fuzz.ratio(title.lower(), title_hit.lower())
        hit['non-arxiv'] = 1 if hit.get('info', {}).get('venue', '') != 'CoRR' else 0 # non-arxiv with priority
        allow_arxiv = True if kwargs.get('allow_arxiv', False) else hit['non-arxiv']
        if hit['fuzzy-score'] >= threshold and allow_arxiv: # directly filter arxiv papers, use arxiv API for this category
            filtered_hits.append(hit)

    if not filtered_hits: # empty
        logger.warning(f'Empty search result with DBLP API for paper `{title}`')
        return None

    sorted_hits = sorted(filtered_hits, key=lambda x: (x['fuzzy-score'], x['non-arxiv']), reverse=True)

    def get_authors(hit):
        authors = []
        # for authors, remove suffix r'\d{4}', e.g., "Kai Yu 0004"
        for auth in hit['info']['authors']['author']:
            name = auth['text'].strip()
            name = re.sub(r' \d{4}$', '', name)
            authors.append(name)
        return authors


    def get_dblp_pdf_url(pdf_url: str) -> str:
        try:
            # Step 1: Check if the link directly points to a PDF
            if pdf_url.endswith('.pdf'):
                return pdf_url
            
            # Step 2: Fetch the webpage content with wget
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
                } # simulate user agent
                response = requests.get(pdf_url, headers=headers, allow_redirects=True, timeout=10)
                response.raise_for_status()
                pdf_url = response.url
                if pdf_url.endswith('.pdf'): return pdf_url
            except Exception as e:
                logger.error(f'Failed to obtain the html file when parsing PDF URL: {pdf_url} withError: {e}')
                return None
    
            # Step 3: Parse the HTML to find PDF link
            # for openreview link: https://openreview.net/forum?id={id} -> forum => pdf
            if pdf_url.startswith('https://openreview.net/forum?'):
                pdf_url = pdf_url.replace('/forum', '/pdf')
                return pdf_url
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            if pdf_url.startswith('https://ojs.aaai.org/'):
                pdf_links = soup.find_all(
                    'a',
                    class_=re.compile(r'\bpdf\b'),
                    href=True
                )
                for link in pdf_links:
                    pdf_link = link['href']
                    # Handle relative links
                    if not pdf_link.startswith('http'):
                        pdf_link = requests.compat.urljoin(pdf_url, pdf_link)
                    return pdf_link
            for link in soup.find_all('a', href=True):
                if link['href'].endswith('.pdf') or '/pdf/' in link['href']:
                    pdf_link = link['href']
                    # Handle relative links
                    if not pdf_link.startswith('http'):
                        pdf_link = requests.compat.urljoin(pdf_url, pdf_link)
                    return pdf_link
            logger.error(f"[Error]: Failed to find the PDF download link from the DBLP URL: {pdf_url}.")
            return None
        
        except Exception as e:
            logger.error(f"[Error]: Unexpected error occurred when finding the PDF download link from the DBLP URL: {pdf_url}.")
            return None


    def get_dblp_bibtex(bibtex_url: str) -> str:
        try:
            if not bibtex_url.endswith('.bib'):
                bibtex_url = bibtex_url + ".bib"
            response = requests.get(bibtex_url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            # logger.error(f"Failed to get the bibtex from URL: {bibtex_url}.")
            return None

    for hit in sorted_hits:
        try:
            title, conference, year = hit['info']['title'], hit['info']['venue'], int(hit['info']['year'])
            volume = hit.get('volume', None)
            # determine the conference abbrevation, conference full name and subfolder for PDF path
            if not conference:
                conference, conference_full, subfolder = None, None, 'uncategorized'
            elif conference.lower() == 'corr':
                conference, conference_full, subfolder = 'arxiv', 'ArXiv', f'arxiv{year}'
            else:
                conference, conference_full = get_ccf_conference_name(conference_abbrev=conference)
                subfolder = conference.lower() + f'{year}' if re.search(r'^[a-z\d]+$', conference.lower()) else 'uncategorized'

            paper_uuid = get_airqa_paper_uuid(title, subfolder)
            if kwargs.get('dataset_dir', None) is not None:
                assert os.path.exists(kwargs['dataset_dir']), f"Invalid dataset directory: {kwargs['dataset_dir']}."
                dataset_dir = kwargs['dataset_dir']
            else: dataset_dir = AIRQA_DIR
            pdf_path = os.path.join(dataset_dir, 'papers', subfolder, f'{paper_uuid}.pdf')
            pdf_url, bibtex = get_dblp_pdf_url(hit['info']['ee']), get_dblp_bibtex(hit['info']['url'])
            if pdf_url is None: # unable to find download link, try the next candidate
                continue
            metadata = {
                "uuid": paper_uuid,
                "title": title,
                "conference": conference,
                "conference_full": conference_full,
                "volume": volume,
                "year": year,
                "authors": get_authors(hit),
                "pdf_url": pdf_url,
                "pdf_path": pdf_path,
                "bibtex": bibtex,
                "abstract": None # not supported yet with DBLP
            }
            return metadata
        except Exception as e:
            logger.error(f'Error occurred when trying to process dblp hit: {json.dumps(hit)}\n{e}')
            pass
    logger.error(f"Failed to extract metadata for paper with title `{title}` and open access PDF in DBLP.")
    return None


def add_ai_research_metadata(
        metadata: Dict[str, Any],
        model: str = 'gpt-4o-mini',
        temperature: float = 0.7,
        tldr_max_length: int = 50,
        tag_number:int = 5,
        **kwargs
    ) -> Dict[str, Any]:
    if kwargs.get("ignore_llm"): return metadata
    if metadata['title'] and metadata['abstract']:
        if not metadata.get('tldr', ""):
            metadata['tldr'] = infer_paper_tldr_from_metadata(pdf_title=metadata['title'],pdf_abstract=metadata['abstract'],max_length=tldr_max_length,model=model,temperature=temperature,**kwargs)
        if not metadata.get('tags', []):
            metadata['tags'] = infer_paper_tags_from_metadata(pdf_title=metadata['title'],pdf_abstract=metadata['abstract'],tag_number=tag_number,model=model,temperature=temperature,**kwargs)
    else:
        logger.error(f"Failed to generate TL;DR and tags for paper {metadata['title']} due to missing title or abstract.")
    return metadata


def write_ai_research_metadata_to_json(metadata: Dict[str, Any], dataset_dir: Optional[str] = None) -> str:
    if dataset_dir is not None:
        assert os.path.exists(dataset_dir), f"Invalid dataset directory: {dataset_dir}."
    else: dataset_dir = AIRQA_DIR
    metadata_path = os.path.join(dataset_dir, 'metadata', f"{metadata['uuid']}.json")
    with open(metadata_path, 'w', encoding='utf8') as of:
        json.dump(metadata, of, indent=4, ensure_ascii=False)
    return metadata_path


def get_ai_research_metadata(
        pdf_path: str,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.0,
        api_tools: List[str] = ['arxiv', 'dblp', 'semantic-scholar'],
        write_to_json: bool = True,
        title_lines: int = 20,
        volume_lines: int = 10,
        tldr_max_length: int = 80,
        tag_number: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
    """ Given input pdf_path, return the metadata of the paper.
    @param:
        pdf_path: str, we support the following inputs:
            - the local path to the PDF file, ending with '.pdf' (if already in {uuid}.pdf format, directly use the uuid)
            - the PDF URL to download the file, e.g., https://arxiv.org/pdf/2108.12212.pdf
            - the uuid of the paper (pre-fetch the metadata collection of conference papers, see utils/airqa_utils.py)
            - the title of the paper (use scholar API calls to get the metadata)
        model and temperature: str, float, the language model and temperature for title inference
        api_tools: List[str], the list of scholar APIs to use, see function `extract_metadata_from_scholar_api`
        **kwargs: Dict[str, Any], other arguments that will be directly passed to the scholar API functions
            - dataset_dir: str, the directory to save the metadata and papers PDF, by default AIRQA_DIR
            - see `extract_metadata_from_scholar_api` for more arguments
    @return: metadata dict (metadata)
        {
            "uuid": "0a02b881-d0b1-59c6-a23e-1feb3bdf4c24", // UUID generated by `get_airqa_paper_uuid`
            "title": "Quantized Side Tuning: Fast and Memory-Efficient Tuning of Quantized Large Language Models",
            "conference_full": "Annual Meeting of the Association for Computational Linguistics", // full title of the conference
            "conference": "ACL", // conference abbreviation
            "year": 2024, // conference year, or which year is this paper published
            "volume": "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)", // volume title or number string
            "bibtex": "...", // bibtex citation text
            "authors": ["Zhengxin Zhang", "Dan Zhao", "Xupeng Miao", ...], // authors list
            "num_pages": 36, // int value
            "pdf_url": "https://aclanthology.org/2024.acl-long.1.pdf", // URL to download the PDF, should end with .pdf
            "pdf_path": "data/dataset/airqa/papers/acl2024/ab14a93a-c5ee-5d60-8713-8b38bd501140.pdf", // local path to save the PDF, rename it with the UUID
            "abstract": "...", // paper abstract text
            "tldr": "...", // brief summary generated by llm based on title and abstract
            "tags": ["..",".."] // paper tags generated by llm based on title and abstract
        }
    """
    metadata_dict = get_airqa_paper_metadata(dataset_dir=kwargs.get('dataset_dir', None))

    # pdf_path: "/path/to/paper/397f31e7-2b9f-5795-a843-e29ea6b28e7a.pdf" -> "397f31e7-2b9f-5795-a843-e29ea6b28e7a"
    if pdf_path.endswith('.pdf') and not pdf_path.startswith('http') and is_valid_uuid(os.path.basename(pdf_path).split('.')[0]):
        pdf_path = os.path.basename(pdf_path).split('.')[0] # extract the uuid of the paper from the filename

    if is_valid_uuid(pdf_path): # is valid paper uuid?
        # [Preferred]: for published conference papers, pre-fetch metadata of all papers
        metadata = metadata_dict.get(pdf_path, {})
        if metadata == {}:
            raise ValueError(f"Metadata for paper UUID {pdf_path} not found locally.")
        else:
            if not (metadata.get('tldr', "") and metadata.get('tags', [])):
                add_ai_research_metadata(metadata=metadata, model=model, temperature=temperature,tldr_max_length=tldr_max_length, tag_number=tag_number,**kwargs)
                if write_to_json: write_ai_research_metadata_to_json(metadata, kwargs.get('dataset_dir', None))
    else:
        if pdf_path.startswith('http') or pdf_path.endswith('.pdf'): # local file path or remote URL
            pdf_path = pdf_path.strip()
            if pdf_path.startswith('http'): # remote URL, download to local path under `tmp_dir` with the same filename
                output_path = os.path.join(TMP_DIR, os.path.basename(pdf_path))
                output_path = download_paper_pdf(pdf_path, output_path)
                if output_path is None:
                    raise ValueError(f"Failed to download the PDF file from {pdf_path}.")
                pdf_path = output_path
            # use LLM to infer the paper title from the first page (we assume the first page MUST contain the title)
            title = infer_paper_title_from_pdf(pdf_path, first_lines=title_lines, model=model, temperature=temperature)
            if title is None:
                raise ValueError(f"Failed to infer the paper title from the first page of the PDF {pdf_path}.")
            logger.info(f"Inferred paper title for {pdf_path} is: {title}")
        else: title = pdf_path

        # use scholar API to get the metadata of the paper
        metadata = extract_metadata_from_scholar_api(title, api_tools, **kwargs)
        if not metadata:
            raise ValueError(f"Failed to extract metadata from the Scholar APIs for the paper with title: {title}.")

        # metadata already exists, just skip
        if metadata["uuid"] in metadata_dict:
            logger.warning(f"Metadata for paper UUID {metadata['uuid']} already exists.")
            metadata = metadata_dict[metadata['uuid']]
            if not (metadata.get('tldr', "") and metadata.get('tags', [])):
                add_ai_research_metadata(metadata=metadata, model=model, temperature=temperature,tldr_max_length=tldr_max_length, tag_number=tag_number,**kwargs)
                if write_to_json: write_ai_research_metadata_to_json(metadata, kwargs.get('dataset_dir', None))
            return metadata

        pdf_path_renamed = metadata['pdf_path']
        if pdf_path.startswith('http') or pdf_path.endswith('.pdf'): # already downloaded the PDF file, rename/move it
            os.makedirs(os.path.dirname(pdf_path_renamed), exist_ok=True)
            os.rename(pdf_path, pdf_path_renamed)
        else: # not downloaded yet
            if not download_paper_pdf(metadata['pdf_url'], pdf_path_renamed):
                raise ValueError(f"Failed to download the PDF file from {metadata['pdf_url']} into {pdf_path_renamed}.")
        metadata['pdf_path'] = get_airqa_relative_path(pdf_path_renamed)
        metadata['num_pages'] = get_num_pages(pdf_path_renamed)
        # if metadata['volume'] is None:
            # metadata['volume'] = infer_paper_volume_from_pdf(pdf_path_renamed, first_lines=volume_lines, last_lines=volume_lines, model=model, temperature=temperature,**kwargs)
        if metadata['abstract'] is None:
            metadata['abstract'] = infer_paper_abstract_from_pdf(pdf_path_renamed, model=model, temperature=temperature,**kwargs)
        metadata_dict[metadata['uuid']] = metadata

        # Generate TL;DR and tags (if abstract is not empty)
        add_ai_research_metadata(metadata, model=model, temperature=temperature, tldr_max_length=tldr_max_length, tag_number=tag_number,**kwargs)

        if write_to_json:
            write_ai_research_metadata_to_json(metadata, kwargs.get('dataset_dir', None))
    return metadata


def aggregate_ai_research_metadata(metadata: Dict[str, Any]) -> List[List[Any]]:
    """ Output:
        [ [ paper_id, title, abstract, num_pages, conference_full, conference_abbreviation, pub_year, volume, download_url, bibtex, authors, pdf_path, tldr, tags ] ]
    """
    result = []
    columns = ["uuid", "title", "abstract", "num_pages", "conference_full", "conference", "year", "volume", "pdf_url", "bibtex", "authors", "pdf_path", "tldr", "tags"]
    defaults = ["", "", "", 0, "", "", 0, "", "", "", [], "", "", []]
    # Extract all metadata
    for i in range(len(columns)):
        result.append(metadata.get(columns[i], defaults[i]))

    result[7] = str(result[7]) # stringify `volume`
    return [result]
#coding=utf8
import json, requests, shutil, uuid, tempfile, subprocess, sys, os, re, logging
from bs4 import BeautifulSoup
from typing import List, Union, Optional, Tuple, Any, Dict
import fitz, pymupdf # PyMuPDF
from fuzzywuzzy import fuzz
import pandas as pd
from urllib.parse import urlencode
from .common_functions import is_valid_uuid, get_uuid, call_llm


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


def download_paper_pdf(pdf_url: str, pdf_path: str) -> Optional[str]:
    """ Download the PDF file from the `pdf_url` into `pdf_path`. Just return the relative `pdf_path` if succeeded.
    """
    if os.path.exists(pdf_path) and os.path.isfile(pdf_path): # PDF file already exists
        logger.warning(f"PDF file {pdf_path} already exists. Just ignore the download from {pdf_url}.")
        return pdf_path
    try:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
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
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    doc.close()
    if num_pages == 0:
        repair_pdf_with_qpdf(pdf_path)
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    return num_pages


def get_ccf_conferences(ccf_file: str = os.path.join(AIRQA_DIR, 'ccf_catalog.csv')) -> pd.DataFrame:
    """ Load the CCF conference dataframe from the provided CSV file.
    """
    return pd.read_csv(ccf_file)


def infer_paper_title_from_pdf(
        pdf_path: str,
        first_lines: Optional[int] = None,
        model: str = 'gpt-4o',
        temperature: float = 0.0 # Use more deterministic decoding with temperature=0.0
    ) -> str:
    """ Use a language model to infer the title of a paper from the top `first_lines` lines of the first page in a PDF.
    """
    doc = fitz.open(pdf_path)
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


def infer_paper_tldr_from_metadata(
        pdf_title: str,
        pdf_abstract: str,
        max_length: int = 60,
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
    """ Use a language model to infer the TL;DR of a paper based on its title and abstract.
    """
    # Call the language model to infer the TL;DR
    template = f"""You are an expert in academic papers. Your task is to write a TL;DR (Too Long; Didn't Read) summary for a research paper based on its title and abstract. The TL;DR should:\n1. Be concise and within {max_length} characters.\n2. Capture the main focus or contribution of the paper.\n3. Be written in a single line without extra formatting or context.\nTitle: {pdf_title}\nAbstract: {pdf_abstract}\nYour response is:
"""
    tldr = call_llm(template, model=model, temperature=temperature, top_p=top_p).strip()

    return tldr


def infer_paper_tags_from_metadata(
        pdf_title: str,
        pdf_abstract: str,
        tag_number: int = 5,
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> list[str]:
    """ Use a language model to infer tags (keywords) of a paper based on its title and abstract.
    """
    # Call the language model to infer the tags
    template = f"""You are an expert in academic papers. Your task is to generate a list of {tag_number} relevant tags (keywords) for a research paper based on its title and abstract. The tags should:\n1. Be concise and relevant to the paper's main focus.\n2. Be unique (avoid duplicates).\n3. Be written as a comma-separated list.\nTitle: {pdf_title}\nAbstract: {pdf_abstract}\nYour response is:
"""
    tags = call_llm(template, model=model, temperature=temperature, top_p=top_p).strip()
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]

    return tag_list


def extract_metadata_from_scholar_api(
        title: str,
        api_tools: List[str] = ['dblp', 'semantic-scholar', 'arxiv'],
        **kwargs
    ) -> Dict[str, Any]:
    """ Given the title of one paper, extract its metadata from provided scholar APIs.
    @param:
        title: str, the title of the paper
        api_tool: List[str], the list of scholar API tools to use, each element is chosen from
            ['dblp', 'semantic-scholar', 'arxiv']
    """
    for tool in api_tools:
        assert tool in ['dblp', 'semantic-scholar', 'arxiv'], f"Invalid scholar API tool: {tool}."
    if not api_tools: # try sequentially with pre-defined orders
        api_tools = ['dblp', 'semantic-scholar', 'arxiv']
    functions = {
        "dblp": dblp_scholar_api,
        "semantic-scholar": semantic_scholar_api,
        "arxiv": arxiv_scholar_api,
    }
    # Call the scholar API to extract the metadata
    metadata_dict = {}
    for tool in api_tools:
        metadata_dict = functions[tool](title, **kwargs)
        if metadata_dict is not None:
            return metadata_dict
    logger.error(f'[Error]: failed to extract the metadata information for paper `{title}` from these provided scholar APIs: {api_tools}.')
    return None


def arxiv_scholar_api(title: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    pass


def semantic_scholar_api(title: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    pass


def dblp_scholar_api(title: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """ Given the title of one paper, extract its metadata from DBLP API.
    @param:
        title: str, the title of the paper
        **kwargs: Dict[str, Any], other arguments that will be directly passed to the DBLP API
            - limit: int, the maximum number of search results to return, by default 10
            - threshold: int, the threshold of the fuzzy ratio to filter the search results, by default 95
            - allow_arxiv: bool, whether to allow arxiv papers in the search results, by default False
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
        title_hit = hit['info'].get('title', '').rstrip('.')
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
                # logger.error(f'Failed to obtain the html file when parsing PDF URL: {pdf_url}')
                return None
    
            # Step 3: Parse the HTML to find PDF link
            # for openreview link: https://openreview.net/forum?id={id} -> forum => pdf
            if pdf_url.startswith('https://openreview.net/forum?'):
                pdf_url = pdf_url.replace('/forum', '/pdf')
                return pdf_url
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                if link['href'].endswith('.pdf') or '/pdf/' in link['href']:
                    pdf_link = link['href']
                    # Handle relative links
                    if not pdf_link.startswith('http'):
                        pdf_link = requests.compat.urljoin(pdf_url, pdf_link)
                    return pdf_link
            # logger.error(f"[Error]: Failed to find the PDF download link from the DBLP URL: {pdf_url}.")
            return None
        
        except Exception as e:
            # logger.error(f"[Error]: Unexpected error occurred when finding the PDF download link from the DBLP URL: {pdf_url}.")
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
            ccf_catalog = get_ccf_conferences()
            # determine the download pdf path
            if not conference:
                conference, conference_full, subfolder = None, None, 'uncategorized'
            elif len(series := ccf_catalog.loc[ccf_catalog.get('abbr').str.lower() == conference.lower(), ['abbr', 'name']]) > 0:
                conference, conference_full = series.iloc[0]
                subfolder = conference.lower() + str(year)
            else:
                subfolder = 'uncategorized'
                conference_full = None
            paper_uuid = get_airqa_paper_uuid(title, subfolder)
            pdf_path = os.path.join(AIRQA_DIR, 'papers', subfolder, f'{paper_uuid}.pdf')
            pdf_url, bibtex = get_dblp_pdf_url(hit['info']['ee']), get_dblp_bibtex(hit['info']['url'])
            if pdf_url is None: # unable to find download link, directly return
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
            # logger.error(f'Error occurred when trying to process hit: {json.dumps(hit)}\n{e}')
            pass

    return


def get_ai_research_metadata(
        pdf_path: str,
        metadata_path: str = os.path.join(AIRQA_DIR, 'uuid2papers.json'),
        model: str = 'gpt-4o',
        temperature: float = 0.0,
        api_tools: List[str] = [],
        write_to_json: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
    """ Given input pdf_path, return the metadata of the paper.
    @param:
        pdf_path: str, we support the following inputs:
            - the local path to the PDF file, ending with '.pdf' (if already in {uuid}.pdf format, directly use the uuid)
            - the PDF URL to download the file, e.g., https://arxiv.org/pdf/2108.12212.pdf
            - the uuid of the paper (pre-fetch the metadata collection of conference papers, see utils/airqa_utils.py)
            - the title of the paper (use scholar API calls to get the metadata)
        metadata_path: str, used to get the metadata of the given paper uuid
        model and temperature: str, float, the language model and temperature for title inference
        api_tools: List[str], the list of scholar APIs to use, see function `extract_metadata_from_scholar_api`
        **kwargs: Dict[str, Any], other arguments that will be directly passed to the scholar API functions
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
            "abstract": "..." // paper abstract text
        }
    """
    if not os.path.exists(metadata_path):
        metadata_dict = {}
    with open(metadata_path, 'r', encoding='utf-8') as inf:
        metadata_dict = json.load(inf)

    # pdf_path: "/path/to/paper/397f31e7-2b9f-5795-a843-e29ea6b28e7a.pdf" -> "397f31e7-2b9f-5795-a843-e29ea6b28e7a"
    if pdf_path.endswith('.pdf') and not pdf_path.startswith('http') and is_valid_uuid(os.path.basename(pdf_path).split('.')[0]):
        pdf_path = os.path.basename(pdf_path).split('.')[0] # extract the uuid of the paper from the filename

    if is_valid_uuid(pdf_path): # is valid paper uuid?
        # [Preferred]: for published conference papers, pre-fetch metadata of all papers
        metadata = metadata_dict.get(pdf_path, {})
        if metadata == {}:
            raise ValueError(f"Metadata for paper UUID {pdf_path} not found in {metadata_path}.")
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
            title = infer_paper_title_from_pdf(pdf_path, first_lines=20, model=model, temperature=temperature)
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
            logger.warning(f"Metadata for paper UUID {metadata['uuid']} already exists in {metadata_path}.")
            return metadata_dict[metadata['uuid']]

        pdf_path_renamed = metadata['pdf_path']
        if pdf_path.startswith('http') or pdf_path.endswith('.pdf'): # already downloaded the PDF file, rename/move it
            os.makedirs(os.path.dirname(pdf_path_renamed), exist_ok=True)
            os.rename(pdf_path, pdf_path_renamed)
        else: # not downloaded yet
            if not download_paper_pdf(metadata['pdf_url'], pdf_path_renamed):
                raise ValueError(f"Failed to download the PDF file from {metadata['pdf_url']} into {pdf_path_renamed}.")
        metadata['pdf_path'] = get_airqa_relative_path(pdf_path_renamed)
        metadata['num_pages'] = get_num_pages(pdf_path_renamed)
        metadata_dict[metadata['uuid']] = metadata

        if write_to_json:
            # new entry added, serialize it
            with open(metadata_path, 'w', encoding='utf8') as of:
                json.dump(metadata_dict, of, indent=4, ensure_ascii=False)
    return metadata


def get_ai_research_tldr_tags(
        metadata: dict,
        model: str = 'gpt-4o',
        tldr_max_length: int = 80,
        tag_number: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
    """ Extend the metadata of a research paper by adding TL;DR and tags using the title and abstract.
    """

    # Extract title and abstract
    uuid = metadata.get("uuid")
    pdf_title = metadata.get("title")
    pdf_abstract = metadata.get("abstract", "")

    metadata_add={"uuid":uuid,
                  "tldr":"",
                  "tags":[]
                }
    # Generate TL;DR and tags if abstract is not empty
    if pdf_title and pdf_abstract:
        tldr = infer_paper_tldr_from_metadata(
            pdf_title=pdf_title,
            pdf_abstract=pdf_abstract,
            max_length=tldr_max_length,
            model=model,
            temperature=temperature,
            top_p =top_p
        )
        tags = infer_paper_tags_from_metadata(
            pdf_title=pdf_title,
            pdf_abstract=pdf_abstract,
            tag_number=tag_number,
            model=model,
            temperature=temperature,
            top_p =top_p
        )

        metadata_add["tldr"] = tldr
        metadata_add["tags"] = tags

    else:
        logger.warning(f"No abstract provided for paper '{pdf_title}', skipping TL;DR and tags generation.")

    return metadata_add


def aggregate_ai_research_metadata(metadata: Dict[str, Any], metadata_add: Dict[str, Any]) -> List[List[Any]]:
    """ Output:
        [ [ paper_id, title, abstract, num_pages, conference_full, conference_abbreviation, pub_year, volume, download_url, bibtex, authors, pdf_path, tldr, tags ] ]
    """
    columns = [ "uuid", "title", "abstract", "num_pages", "conference_full", "conference", "year", "volume", "pdf_url", "bibtex", "authors", "pdf_path" ]
    defaults = [ "", "", "", 0, "", "", 0, "", "", "", [], ""]
    result = []

    # Extract base metadata
    for i in range(len(columns)):
        result.append(metadata.get(columns[i], defaults[i]))

    # Add TL;DR and tags from metadata_add
    result.append (metadata_add.get("tldr", ""))  # Add TL;DR
    result.append (metadata_add.get("tags", []))  # Add tags

    result[7] = str(result[7]) # stringify `volume`
    return [result]
#coding=utf8
"""
    Merely used for testing_domain, a simple database for debugging.
    Note that, database-specific pipeline functions and aggregate functions should be defined in the corresponding file under functions/ directory.
    And we suggest naming the function with some convention, e.g., aggregate_{database_name}_table_{table_name}
"""
from typing import List, Dict, Union, Any


def aggregate_test_domain_table_pdf_meta(pdf_output: Dict[str, Union[str, List[str]]]) -> List[List[Any]]:
    """ Aggregate the test domain table PDF meta data.
    @args:
        pdf_output: Dict[str, Union[str, List[str]]], the output dictionary containing the following keys:
            - pdf_id: str, the UUID of the PDF file.
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
            - page_contents: List[str], the list of strings, each string represents the content of each page.
            - page_uuids: List[str], the list of UUIDs for each page if generate_uuid is True.
    @return:
        table: List[List[Any]], the table containing the following columns sequentially:
            - pdf_id: str, the UUID of the PDF file.
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
    """
    return [[pdf_output['pdf_id'], pdf_output['pdf_name'], pdf_output['pdf_path']]]



def aggregate_test_domain_table_pdf_pages(pdf_output: Dict[str, Union[str, List[str]]], page_output: Dict[str, Union[List[str], str]]) -> List[List[Any]]:
    """ Aggregate the test domain table PDF content.
    @args:
        pdf_output: Dict[str, Union[str, List[str]]], the output dictionary containing the following keys:
            - pdf_id: str, the UUID of the PDF file.
            - pdf_name: str, the name of the PDF file.
            - pdf_path: str, the path to the PDF file.
            - page_contents: List[str], the list of strings, each string represents the content of each page.
            - page_uuids: List[str], the list of UUIDs for each page if generate_uuid is True.
        page_output: Dict[str, Union[List[str], str]], the output dictionary containing the following keys:
            - text_summary: List[str], the list of text summaries for each page.
    @return:
        table: List[List[Any]], the table containing the following columns sequentially:
            - page_id: str, the UUID of the page, primary key
            - page_number: int, the page number
            - page_content: str, the content of the page
            - page_summary: str, the summary of the page content
            - pdf_id: str, foreign key, the UUID of the PDF file
    """
    results = []
    for idx, (page_uuid, page_content, page_summary) in enumerate(zip(pdf_output['page_uuids'], pdf_output['page_contents'], page_output['text_summary'])):
        results.append([page_uuid, idx + 1, page_content, page_summary, pdf_output['pdf_id']])
    return results
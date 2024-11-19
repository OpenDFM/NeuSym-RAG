#coding=utf8

from .common_functions import (
    get_uuid,
)

from .pdf_functions import (
    get_pdf_page_text,
    get_text_summary
)

from .biology_paper import (
    get_biology_paper_per_page_uuid,
    get_biology_paper_per_page_content_uuid,
    get_biology_paper_per_page_chunk_uuid_and_text,
    aggregate_biology_paper_table_metadata,
    aggregate_biology_paper_table_pages,
    aggregate_biology_paper_table_content_types,
    aggregate_biology_paper_table_content,
    aggregate_biology_paper_table_chunks,
    aggregate_biology_paper_table_parent_child_relations
)

from .financial_report import (
    get_financial_report_per_page_content_uuid,
    get_financial_report_per_page_uuid,
    get_financial_report_per_page_chunk_uuid_and_text,
    aggregate_financial_report_table_metadata,
    aggregate_financial_report_table_pages,
    aggregate_financial_report_table_content,
    aggregate_financial_report_table_chunks,
    get_financial_report_per_page_tableinpages,
    get_financial_report_per_page_tableinpages_uuid,
    aggregate_financial_report_table_tableinpages
)

from .ai_research import (
    get_ai_research_per_page_figure_uuid_and_summary,
    # aggregate_ai_research_table_figures
)

from .test_domain import (
    aggregate_test_domain_table_pdf_meta,
    aggregate_test_domain_table_pdf_pages
)

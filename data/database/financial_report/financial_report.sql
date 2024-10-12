/* database financial_report: This database is designed to store and organize data from annual financial reports, including metadata, tables, and sections extracted from reports such as FORM 10-K and non-standard format reports.
*/
/* table metadata: This table stores metadata about each report, including the number of pages, report path and report id.
*/
CREATE TABLE IF NOT EXISTS metadata (
	report_id UUID, -- A unique identifier for each report.
	report_pages INTEGER, -- The number of pages in this report.
	report_path VARCHAR, -- The path to the PDF file.,
	PRIMARY KEY (report_id)
);
/* table pages: This table stores page content of the reports, including their content and their order within the report.
*/
CREATE TABLE IF NOT EXISTS pages (
	page_id UUID, -- A unique identifier for each page within a report.
	page_number INTEGER, -- The page number on which this section is located in the report.
	page_width INTEGER, -- The pixel width of the current page.
	page_height INTEGER, -- The pixel height of the current page.
	page_content VARCHAR, -- The content of the page.
	ref_report_id UUID, -- A foreign key linking to the report ID in the metadata table.,
	PRIMARY KEY (page_id),
	FOREIGN KEY (ref_report_id) REFERENCES metadata(report_id)
);
/* table content: This table contains concrete content information for each recognized element in each page in one financial report.
*/
CREATE TABLE IF NOT EXISTS content (
	content_id UUID, -- A unique identifier for each content element.
	text_content VARCHAR, -- The text content of the current element.
	bounding_box INTEGER[4], -- The bounding box of the current element in the format [x0, y0, w, h], where (x0, y0) represents the coordinates of the top-left corner and (w, h) represents the width and height which are used to determine the shape of the rectangle.
	ordinal INTEGER, -- Each content element is labeled with one distinct integer number in the current page, which starts from 0.
	ref_report_id UUID, -- A foreign key linking to the report ID in the `metadata` table.
	ref_page_id UUID, -- A foreign key linking to the page ID in the `pages` table.,
	PRIMARY KEY (content_id),
	FOREIGN KEY (ref_report_id) REFERENCES metadata(report_id),
	FOREIGN KEY (ref_page_id) REFERENCES pages(page_id)
);
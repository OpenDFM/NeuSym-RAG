/* database test_domain: This database contains test data for debugging purpose.
*/
/* table pdf_meta: Record the metadata of one PDF file.
*/
CREATE TABLE IF NOT EXISTS pdf_meta (
	pdf_id UUID, -- A unique identifier for each PDF file.
	pdf_name VARCHAR, -- The name of the PDF file.
	pdf_path VARCHAR, -- The file path to the original PDF file.,
	PRIMARY KEY (pdf_id)
);
/* table pdf_pages: Record the content of each PDF page.
*/
CREATE TABLE IF NOT EXISTS pdf_pages (
	page_id UUID, -- A unique identifier for each page in one pdf.
	page_number INTEGER, -- Page number, starting from 1 to the maximum page counts.
	page_content VARCHAR, -- Extracted text content of one page.
	page_summary VARCHAR, -- A brief summary of the current page, less than 50 words.
	pdf_id UUID, -- A foreign key linking to the PDF ID in the metadata table.,
	PRIMARY KEY (page_id),
	FOREIGN KEY (pdf_id) REFERENCES pdf_meta(pdf_id)
);
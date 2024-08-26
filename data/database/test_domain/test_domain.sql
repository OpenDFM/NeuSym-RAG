CREATE TABLE IF NOT EXISTS pdf_meta (
	pdf_id UUID,
	pdf_name VARCHAR,
	pdf_path VARCHAR,
	PRIMARY KEY (pdf_id)
);
CREATE TABLE IF NOT EXISTS pdf_pages (
	page_id UUID,
	page_number INTEGER,
	page_content VARCHAR,
	page_summary VARCHAR,
	pdf_id UUID,
	PRIMARY KEY (page_id),
	FOREIGN KEY (pdf_id) REFERENCES pdf_meta(pdf_id)
);
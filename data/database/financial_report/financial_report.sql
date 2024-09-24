CREATE TABLE IF NOT EXISTS metadata (
	report_id UUID,
	report_pages INTEGER,
	report_type VARCHAR,
	report_path VARCHAR,
	PRIMARY KEY (report_id)
);
CREATE TABLE IF NOT EXISTS pages (
	page_id UUID,
	page_number INTEGER,
	report_id UUID,
	page_description VARCHAR,
	page_content VARCHAR,
	PRIMARY KEY (page_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id)
);
CREATE TABLE IF NOT EXISTS sections (
	section_id UUID,
	report_id UUID,
	section_name VARCHAR,
	section_content VARCHAR,
	PRIMARY KEY (section_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id)
);
CREATE TABLE IF NOT EXISTS tables (
	table_id UUID,
	report_id UUID,
	ref_section UUID,
	table_name VARCHAR,
	table_description VARCHAR,
	ref_page UUID,
	table_content VARCHAR,
	PRIMARY KEY (table_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id),
	FOREIGN KEY (ref_section) REFERENCES sections(section_id),
	FOREIGN KEY (ref_page) REFERENCES pages(page_id)
);

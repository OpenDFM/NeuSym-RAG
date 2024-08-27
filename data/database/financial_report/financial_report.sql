CREATE TABLE IF NOT EXISTS metadata (
	report_id UUID,
	company_name VARCHAR,
	report_type VARCHAR,
	report_pages INTEGER,
	fiscal_year INTEGER,
	source_url VARCHAR,
	PRIMARY KEY (report_id)
);
CREATE TABLE IF NOT EXISTS sections (
	section_id UUID,
	report_id UUID,
	section_name VARCHAR,
	section_tag VARCHAR,
	ref_table UUID,
	section_page INTEGER,
	section_content VARCHAR,
	PRIMARY KEY (section_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id)
);
CREATE TABLE IF NOT EXISTS sub_sections (
	sub_section_id UUID,
	ref_section UUID,
	sub_section_name VARCHAR,
	sub_section_tag VARCHAR,
	sub_section_page INTEGER,
	sub_section_content VARCHAR,
	PRIMARY KEY (sub_section_id),
	FOREIGN KEY (ref_section) REFERENCES sections(section_id)
);
CREATE TABLE IF NOT EXISTS tables (
	table_id UUID,
	report_id UUID,
	ref_section UUID,
	table_name VARCHAR,
	description VARCHAR,
	table_page INTEGER,
	row_count INTEGER,
	column_count INTEGER,
	table_content VARCHAR,
	PRIMARY KEY (table_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id),
	FOREIGN KEY (ref_section) REFERENCES sections(section_id)
);

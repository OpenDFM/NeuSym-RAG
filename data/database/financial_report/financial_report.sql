CREATE TABLE IF NOT EXISTS metadata (
	report_id UUID,
	company_name VARCHAR,
	report_type VARCHAR,
	fiscal_year INTEGER,
	source_url VARCHAR,
	PRIMARY KEY (report_id)
);
CREATE TABLE IF NOT EXISTS sections (
	section_id UUID,
	report_id UUID,
	section_name VARCHAR,
	section_type VARCHAR,
	ref_table UUID,
	section_order INTEGER,
	section_content VARCHAR,
	PRIMARY KEY (section_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id)
);
CREATE TABLE IF NOT EXISTS tables (
	table_id UUID,
	report_id UUID,
	ref_section UUID,
	table_name VARCHAR,
	description VARCHAR,
	table_order INTEGER,
	row_count INTEGER,
	column_count INTEGER,
	table_content VARCHAR,
	PRIMARY KEY (table_id),
	FOREIGN KEY (report_id) REFERENCES metadata(report_id),
	FOREIGN KEY (ref_section) REFERENCES sections(section_id)
);
CREATE TABLE IF NOT EXISTS metadata (
	paper_id VARCHAR,
	num_pages INTEGER,
	local_folder VARCHAR,
	PRIMARY KEY (paper_id)
);
CREATE TABLE IF NOT EXISTS pages (
	page_id UUID,
	figure_path VARCHAR,
	page_number INTEGER,
	page_width INTEGER,
	page_height INTEGER,
	page_content VARCHAR,
	ref_paper_id VARCHAR,
	PRIMARY KEY (page_id),
	FOREIGN KEY (ref_paper_id) REFERENCES metadata(paper_id)
);
CREATE TABLE IF NOT EXISTS content_types (
	type_id INTEGER,
	content_type_name VARCHAR,
	PRIMARY KEY (type_id)
);
CREATE TABLE IF NOT EXISTS content (
	content_id UUID,
	content_type INTEGER,
	text_content VARCHAR,
	bounding_box INTEGER[4],
	ordinal INTEGER,
	ref_paper_id VARCHAR,
	ref_page_id UUID,
	PRIMARY KEY (content_id),
	FOREIGN KEY (content_type) REFERENCES content_types(type_id),
	FOREIGN KEY (ref_paper_id) REFERENCES metadata(paper_id),
	FOREIGN KEY (ref_page_id) REFERENCES pages(page_id)
);
CREATE TABLE IF NOT EXISTS parent_child_relations (
	parent_id UUID,
	child_id UUID,
	PRIMARY KEY (parent_id, child_id),
	FOREIGN KEY (parent_id) REFERENCES content(content_id),
	FOREIGN KEY (child_id) REFERENCES content(content_id)
);
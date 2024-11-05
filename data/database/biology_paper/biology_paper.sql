/* database biology_paper: This database contains data about each biology paper. Each PDF file is represented or parsed via different views, e.g., pages, sections, figures, tables, and references. We also extract the concrete content inside each concrete element via OCR.
*/
/* table metadata: This table stores metadata about each report, including the number of pages, local folder and paper id.
*/
CREATE TABLE IF NOT EXISTS metadata (
	paper_id VARCHAR, -- A unique identifier for each paper.
	num_pages INTEGER, -- The total number of pages in the biology paper.
	local_folder VARCHAR, -- The local folder where the PDF file is stored, and each page is stored as a separate image with the complete path {local_folder}/{paper_id}_{page_number}.png.,
	PRIMARY KEY (paper_id)
);
/* table pages: This table stores the concrete information for each page.
*/
CREATE TABLE IF NOT EXISTS pages (
	page_id UUID, -- A unique identifier for each page within a paper.
	page_path VARCHAR, -- Path to the current page (each PDF page is stored as a .png screenshot file).
	page_number INTEGER, -- The page number on which this table is located in the paper.
	page_width INTEGER, -- The pixel width of the current page.
	page_height INTEGER, -- The pixel height of the current page.
	page_content VARCHAR, -- The full text content of the current page, including all main texts, section titles, references, tables, and figures.
	ref_paper_id VARCHAR, -- A foreign key linking to the paper ID in the metadata table.,
	PRIMARY KEY (page_id),
	FOREIGN KEY (ref_paper_id) REFERENCES metadata(paper_id)
);
/* table content_types: This table defines the type id for each of the following content types: 1 -> main text, 2 -> section title, 3 -> reference list, 4 -> tables in the paper, 5 -> figures in the paper.
*/
CREATE TABLE IF NOT EXISTS content_types (
	type_id INTEGER, -- A unique identifier for each content type, only supporting 1 - 5.
	content_type_name VARCHAR, -- The name of the content type.,
	PRIMARY KEY (type_id)
);
/* table content: This table contains concrete content information for each recognized element in each page in one biology paper.
*/
CREATE TABLE IF NOT EXISTS content (
	content_id UUID, -- A unique identifier for each content element.
	content_type INTEGER, -- The type id of content, linking to the content_types table.
	text_content VARCHAR, -- The text content of the current element.
	bounding_box INTEGER[4], -- The bounding box of the current element in the format [x0, y0, w, h], where (x0, y0) represents the coordinates of the top-left corner and (w, h) represents the width and height which are used to determine the shape of the rectangle.
	ordinal INTEGER, -- Each content element is labeled with one distinct integer number in the current page, which starts from 0.
	ref_paper_id VARCHAR, -- A foreign key linking to the paper ID in the `metadata` table.
	ref_page_id UUID, -- A foreign key linking to the page ID in the `pages` table.,
	PRIMARY KEY (content_id),
	FOREIGN KEY (content_type) REFERENCES content_types(type_id),
	FOREIGN KEY (ref_paper_id) REFERENCES metadata(paper_id),
	FOREIGN KEY (ref_page_id) REFERENCES pages(page_id)
);
/* table chunks: This table contains the text content of each chunk of text (chunk size = 512 tokens with no overlapping) in each page in one biology paper. A chunk is a sub-text that is extracted from the main text, such as a sentence or a paragraph.
*/
CREATE TABLE IF NOT EXISTS chunks (
	chunk_id UUID, -- A unique identifier for each chunk of text.
	text_content VARCHAR, -- The text content of the current chunk.
	ordinal INTEGER, -- Each chunk is labeled with one distinct integer number in the current page, which starts from 0.
	ref_paper_id VARCHAR, -- A foreign key linking to the paper ID in the `metadata` table.
	ref_page_id UUID, -- A foreign key linking to the page ID in the `pages` table.,
	PRIMARY KEY (chunk_id),
	FOREIGN KEY (ref_paper_id) REFERENCES metadata(paper_id),
	FOREIGN KEY (ref_page_id) REFERENCES pages(page_id)
);
/* table parent_child_relations: This table defines the parent-child relationships between different content elements, for example, some main texts belong to one section title.
*/
CREATE TABLE IF NOT EXISTS parent_child_relations (
	parent_id UUID, -- The parent content element ID.
	child_id UUID, -- The child content element ID.,
	PRIMARY KEY (parent_id, child_id),
	FOREIGN KEY (parent_id) REFERENCES content(content_id),
	FOREIGN KEY (child_id) REFERENCES content(content_id)
);
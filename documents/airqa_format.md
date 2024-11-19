# PDF-based Question Answering on Artificial Intelligence Research Papers (AIR-QA)

## Folder Structure

```txt
- data/dataset/airqa/ # root folder
    - test_data.jsonl # JSON line file where each line represents one test data
    - data_format.json.template # template file for each data point
    - uuid2papers.json # it stores the mapping from paper UUID to the metadata of each file (dict), like title, conference, year, authors, etc.
    - errata.json # fix metadata errors for some papers
    - papers/ # stores all PDF files of papers, organized by lowercase {conference}{year} sub-folder and renamed by paper UUIDs
        - acl2024/
            - 0a1e5410-f9f1-5877-ae3a-151c214762e1.pdf
            - ...
        - acl2023/
            - ...
        ...
    - processed_data/ # folder used to store pre-processed result for each PDF file, e.g., extracted images or LLM-generated page summaries
        - 0a1e5410-f9f1-5877-ae3a-151c214762e1.json
    - examples/ # during annotation, each data is stored as a separate file ${question_uuid}.json following `data_format.json.template`
        - 0dadc5c6-a5f7-572b-9a20-fc9b907eddb9.json
```


## AIR-QA Tags

This section describes different question categories (or tags) for classification.

### Category 1: Task Goals

- `paper retrieval`: retrieve papers with constraints, e.g.,
    - papers published by a specific author or institute
- `single-doc details`: ask technical details of one single paper, e.g.,
    - list 3 major contributions of this work
- `multi-doc analysis`: involve multiple papers, may require comparison, calculation, aggregation and multi-step reasoning, e.g.,
    - which agent framework is better on this popular benchmark
- `metadata query`: query metadata about papers (note that, here metadata can be obtained in raw PDF file), e.g.,
    - whether the code is public available
- `comprehensive q&a`: if the task does not belong to any category above or requires integration of multiple task types, e.g.,
    - which author published the most papers regarding multi-modal llm pre-training in acl 2023 (matadata query + paper retrieval)


### Category 2: Key Capabilities


### Category 3: Evaluation Methods


### Category 4: Hardness

TODO: will be automatically determined by the number of reasoning steps in the annotated `.json` data (field `reasoning_steps`)

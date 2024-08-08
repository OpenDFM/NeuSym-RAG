# Project for RAG-Scheduler


<p align="center">
  <img src="assets/rag-framework.png" alt="our-framework">
</p>

Main contribution:

- Leverage an RAG scheduler module to combine different retrieval methods (both neural-based unstructured retrieval and symbolic-based structured retrieval)
- Salient features are:
  1. Retrieval workflow orchestration, which includes:
    - decompose the original query into chain of retrievals (plan-and-solve)
    - reason upon the execution results or temporary retrieval, and determine the next action (execute-and-act)
    - decide when to terminate the retrieval (when-to-stop)
    - combine the chain-of-retrieval results (summarize-and-reflect)
    - in a word, it treats the retrieval process as an **agentic** task
  2. Retrieval model selection and external tools invoke
    - classic vector-based neural retrieval
    - specialized text-to-SQL symbolic retrieval
    - external tools or API, e.g., Web search and PDF parsing utility


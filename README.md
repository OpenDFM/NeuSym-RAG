# Project for Retrieval Scheduler


<p align="center">
  <img src="assets/rag-framework.png" alt="our-framework">
</p>

Main contribution:

- Leverage a retrieval scheduler module to combine different retrieval methods:
  - vector-based neural retrieval
  - text-to-SQL symbolic retrieval
  - external tools, e.g., Web search, PDF parsing, image captioning, etc.
- Treat retrieval as an **agentic** task, salient features include:
  1. Retrieval module and external tools selection
  2. Retrieval workflow orchestration, which includes:
      - decompose the original query into chain of retrievals (plan-and-solve)
      - reason upon temporary retrieval results, and determine the next action (interaction)
      - decide when to terminate (when-to-stop)
      - combine the chain-of-retrieval results (summarize-and-reflect)
  3. Plug-and-play retrieval module extension


## Documents and Tutorials

The documents for this project and fine-grained topics are discussed in the folder `documents/`. The checklist includes:

- [`documents/database.md`](documents/database.md):
  - How to define database schema and its format;
  - How to fill in database content with generic Python class `DatabasePopulation` and module `utils.functions`;
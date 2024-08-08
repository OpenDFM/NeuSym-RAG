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
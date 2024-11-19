#coding=utf8
""" Encourage the LLM to interact with the environment in multi-turn before making the final judgment.
Note that, these hint prompts are only used in an interactive environment.
"""

TEXT2SQL_HINT_PROMPT = """
#### Suggestions or Hints for Agent Interaction

1. Explore database content carefully. Before writing the final SQL query, examine the database schema, cell formats and value distribution to ensure accuracy. Pay special attention to string match conditions to avoid errors caused by differences in letter case, whitespace, or morphological variations.

2. Iterate and refine:
- If SQL execution result is not satisfactory, try another SQL query to visit different table/columns or check the SQL values.
- Use your findings to validate or enrich the final response.

3. Ensure confidence. That is, only make a final decision when you are confident that the retrieved information fully addresses the user’s query.
"""

TEXT2VEC_HINT_PROMPT = """
#### Suggestions or Hints for Agent Interaction

1. Explore multiple retrieval strategies. For example:
- Query various embedding models (collections) to find the most relevant context.
- Experiment with different encodable (table, column) pairs to extract diverse types of information.

2. Iterate and refine:
- If the initial retrieval is insufficient, try alternative approaches or parameter settings.
- Use your findings to validate or enrich the final response.

3. Ensure confidence. That is, only make a final decision when you are confident that the retrieved information fully addresses the user’s query.
"""

HYBRDI_RAG_HINT_PROMPT = """
#### Suggestions or Hints for Agent Interaction

1. Explore multiple retrieval strategies. For example:
- Experiment with different (table, column) pairs to extract diverse types of information.
- Query various embedding models (collections) to find the most relevant context.

2. Combine both structured and unstructured data. Concretely:
- Use SQL queries to retrieve precise facts and structured data. Pay special attention to morphological variations in cell values.
- Perform similarity searches in the vectorstore to capture semantic relationships and hidden insights.

3. Iterate and refine:
- If SQL execution result is not satisfactory, try alternative SQL queries to explore the database content carefully.
- If the vector-based neural retrieval is insufficient, try alternative approaches or parameter settings.
- Use your findings to validate or enrich the final response.

4. Ensure confidence. That is, only make a final decision when you are confident that the retrieved information fully addresses the user’s query.
"""
TEXT2SQL_ACTION_AND_OBSERVATION_PROMPT = f"""## Action and Observation Space

The action space that you can take includes two types of actions: `GenerateSQL` and `GenerateAnswer`.

### Action Type 1

GenerateSQL:
```sql
SELECT ...
```

Description: Generate an SQL query to retrieve the desired information from the database. Please refer to the concrete database schema to produce the valid SQL. Remember to replace the code within sql backticks with your executable SQL program.

Observation: The observation space is the execution result of the SQL query. You do not need to worry about the actual execution, we will perform it for you. If the SQL failed to execute, we will return the error message. And extremely long SQL output will be truncated for further actions.

### Action Type 2

GenerateAnswer:
```txt
a single answer string or multiple lines that conform to the output format
```

Description: When you decide to take this action, we assume that the retrieved results suffice to answer the user question. Please strictly adhere to the output format. And remember to replace the content within text backticks with your answer.

Observation: There is no observation space for this action, since it indicates the completion of the task and end of the interaction.
"""
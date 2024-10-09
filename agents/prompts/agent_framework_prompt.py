#coding=utf8
from agents.prompts.action_and_observation_prompt import *
from agents.prompts.system_prompt import *


TEXT2SQL_REACT_PROMPT = f"""{TEXT2SQL_BASELINE_SYSTEM_PROMPT}

{TEXT2SQL_ACTION_AND_OBSERVATION_PROMPT}

## React Interaction Framework

The complete interaction procedure proceeds like this:

----

Question: Is there any ...?
Answer Format: it describes the format of the final answer, e.g., the answer is "Yes" or "No" without punctuation.
Database Schema:
... # detailed serialized database schema

Thought: your reasoning process, why we take this action ...
Action: GenerateSQL:
```sql
SELECT ... FROM ... WHERE ...
```
Observation: Output table or error message of the SQL execution.

Thought: further reasoning process if needed ...
Action: GenerateSQL:
```sql
SELECT ...
```
Observation: Further execution results.

... # interaction with the database may have zero or multiple turns

Thought: final reasoning process to produce the final answer ...
Action: GenerateAnswer:
```txt
Yes
```

----

In general, the main interaction loop consists of an interleaved of triplets (Thought, Action, Observation), except the last `GenerateAnswer` action which does not have "Observation". Now, let's start!
"""
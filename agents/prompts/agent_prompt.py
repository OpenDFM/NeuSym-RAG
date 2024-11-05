#coding=utf8

REACT_PROMPT = """{system_prompt}

--------

{action_space_prompt}

--------

## Interaction Framework

The main interaction procedure proceeds like this:

----

[Thought]: ... reasoning process, why to take this action ...
[Action]: ... which action to take, please strictly conform to the action specification ...
[Observation]: ... execution results or error message after taking the action ...

... # more interleaved triplets of ([Thought], [Action], [Observation])

[Thought]: ... reasoning process to produce the final answer ...
[Action]: ... the terminate action `GenerateAnswer`, there is no further observation ...

----

In general, the main interaction loop consists of an interleaved of triplets ([Thought], [Action], [Observation]), except the last `GenerateAnswer` action which does not have "[Observation]:". Remember that, for each question, you only have {max_turn} interaction turns at most. Now, let's start!
"""

TWO_STAGE_TEXT2SQL_PROMPT = [
"""{system_prompt}

--------

[Question]: {question}
[Database Schema]: {database_schema}

You can firstly give your reasoning process, followed by the SQL query in the following format (REMEMBER TO WRAP YOUR SQL IN THREE BACKTICKS):

```sql\nconcrete sql query\n```
"""

,

"""{system_prompt}

--------

[Question]: {question}
[Context]: {context}
[Answer Format]: {answer_format}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER IN THREE BACKTICKS):

```txt\nfinal answer\n```
"""
]

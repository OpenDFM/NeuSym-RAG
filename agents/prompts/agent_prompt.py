#coding=utf8

TRIVIAL_PROMPT = """ {system_prompt}

--------

Here is the task input:

[Question]: {question}
[Retrieved Context]: {context}
[Answer Format]: {answer_format}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER WITH REQUIRED FORMAT IN THREE BACKTICKS):

```txt\nfinal answer\n```
"""


CLASSIC_RAG_PROMPT = """ {system_prompt}

--------

Here is the task input:

[Question]: {question}
[Retrieved Context]: {context}
[Answer Format]: {answer_format}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER WITH REQUIRED FORMAT IN THREE BACKTICKS):

```txt\nfinal answer\n```
"""


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

In general, the main interaction loop consists of an interleaved of triplets ([Thought], [Action], [Observation]), except the last `GenerateAnswer` action which does not have "[Observation]:". You need to predict the "[Thought]: ..." followed by the "[Action]: ..." for each turn, and we will execute your action in the environment and provide the "[Observation]: ..." for the previous action.

--------
{hint_prompt}

Remember that, for each question, you only have {max_turn} interaction turns at most. Now, let's start!
"""


TWO_STAGE_TEXT2SQL_PROMPT = [
"""{system_prompt}

--------

[Question]: {question}
{pdf_context}
[Database Schema]: {database_schema}

You can firstly give your reasoning process, followed by the SQL query in the following format (REMEMBER TO WRAP YOUR SQL IN THREE BACKTICKS):

```sql\nconcrete sql query\n```

Here is your response:
"""

,

"""{system_prompt}

--------

[Question]: {question}
{pdf_context}
[SQL]: {sql}
[Retrieved Context]: {context}
[Answer Format]: {answer_format}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER IN THREE BACKTICKS):

```txt\nfinal answer\n```

Here is your response:
"""
]


TWO_STAGE_TEXT2VEC_PROMPT = [
"""{system_prompt}

--------

Here is a detailed specification of the `RetrieveFromVectorstore` action that you need to predict:

{action_prompt}

--------

The task input is:

[Question]: {question}
{pdf_context}
[Vectorstore Schema]:
{vectorstore_schema}

You can firstly give your reasoning process, followed by the `RetrieveFromVectorstore` action with pre-defined JSON format (REMEMBER TO WRAP YOUR ACTION IN THREE BACKTICKS):

```json\nconcrete JSON-format action\n```

Here is your response:
"""

,

"""{system_prompt}

--------

[Question]: {question}
{pdf_context}
[Retrieved Context]:\n{context}
[Answer Format]: {answer_format}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER IN THREE BACKTICKS):

```txt\nfinal answer\n```

Here is your response:
"""
]
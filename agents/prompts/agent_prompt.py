#coding=utf8


CLASSIC_RAG_AGENT_PROMPT = """{system_prompt}

--------

Here is the task input:

{task_input}
{context}

You can firstly give your reasoning process, followed by the final answer in the following format (REMEMBER TO WRAP YOUR ANSWER WITH REQUIRED FORMAT IN THREE BACKTICKS):

```txt\nfinal answer\n```
"""


ITERATIVE_RAG_AGENT_PROMPT = """{system_prompt}

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


TWO_STAGE_SYM_RAG_AGENT_PROMPT = [
"""{system_prompt}

--------

{task_input}

You can firstly give your reasoning process, followed by the SQL query in the following format (REMEMBER TO WRAP YOUR SQL IN THREE BACKTICKS):

```sql\nconcrete sql query\n```

Remember that, for each question, you only have one chance to predict the SQL. And your response is:
""", CLASSIC_RAG_AGENT_PROMPT]


TWO_STAGE_NEU_RAG_AGENT_PROMPT = [
"""{system_prompt}

--------

Here is a detailed specification of the `RetrieveFromVectorstore` action that you need to predict:

{action_space_prompt}

--------

{task_input}

You can firstly give your reasoning process, followed by the `RetrieveFromVectorstore` action with pre-defined JSON format (REMEMBER TO WRAP YOUR ACTION IN THREE BACKTICKS):

```json\nconcrete JSON-format action\n```

Remember that, for each question, you only have one chance to predict the `RetrieveFromVectorstore` action. And your response is:""", CLASSIC_RAG_AGENT_PROMPT]


TWO_STAGE_HYBRID_RAG_AGENT_PROMPT = [
"""{system_prompt}

--------

{action_space_prompt}

--------

{task_input}

You can firstly give your reasoning process, followed by the `RetrieveFromVectorstore` or `RetrieveFromDatabase` action with pre-defined JSON format (REMEMBER TO WRAP YOUR ACTION IN THREE BACKTICKS):

```json\nconcrete JSON-format action\n```

Remember that, for each question, you can ONLY predict one action and ONLY have one chance. And your response is:""", CLASSIC_RAG_AGENT_PROMPT]
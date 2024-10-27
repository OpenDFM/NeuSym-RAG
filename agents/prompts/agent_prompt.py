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

TWOSTAGE_PROMPT = [

"""{system_prompt}

--------

Generate an SQL query to retrieve the desired information from the database. Please refer to the concrete database schema to produce the valid SQL. Output in the following format:

----

[Thought]: ... reasoning process, why to generate this SQL ...
[Action]: GenerateSQL:\n```sql\nconcrete sql query\n```
[Observation]: ... The observation space is the execution result of the SQL query. You do not need to worry about the actual execution, we will perform it for you. If the SQL failed to execute, we will return the error message. And extremely long SQL output will be truncated for further actions. ...

----
"""

,


"""{system_prompt}

--------

## Interaction Framework

Generate the final answer based on the information retrieved. Please strictly adhere to the answer format of the current question. Output in the following format:

----

[Thought]: ... reasoning process to produce the final answer ...
[Action]: GenerateAnswer:\n```txt\nfinal answer\n```

----

Remember that, even you didn\'t retrieve the desired information in the previous SQL query, you still need to generate the final answer based on the current observation. DO NOT GENERATE SQL QUERY ANYMORE.
"""
]
#coding=utf8

EXPLORE_PROMPT = """You are an intelligent annotation system who is expert in posing questions. 

You will be given an AI research paper, and your task is to generate a question based on {description}. Your output should be in the following format:
```txt
[Question]: Your question here.
[Answer]: Your answer here.
[Reasoning Steps]: Your reasoning steps here.
```
Notice that:
1. Remember to wrap the question and answer with triple backticks.
2. Don't include the answer in the question or in the reasoning steps.
3. Your question should be as objective as possible.
4. Your answer should be concise and clear.
5. Your reasoning steps should serve as a hint to the question. Focus on the action that the participant should do to find the answer, e.g. "Locate the section that describes ...", "Find the name of the model in the section.", but avoid including specific contexts, numbers, titles or captions.
{hint}

Let's think step-by-step, and then provide the final question and answer.

{context}

{image}
"""

DESCRIPTION_PROMPT = {
"text": "the content of the section in MARKDOWN format"
}

CONTEXT_PROMPT = {
"text": """The content of the section is as follows:

```markdown
{context}
```""",
"table": ""
}

HINT_PROMPT = {
"text": ""
}

IMAGE_PROMPT = """
"""
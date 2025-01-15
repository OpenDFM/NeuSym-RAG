#coding=utf8

_EXPLORE_PROMPT = """You are an intelligent annotation system who is expert in posing questions. 

You will be given an AI research paper, and your task is to generate a question based on {description}. Your output should be in the following format:
```txt
[Question]: Your question here.
[Answer]: Your answer here.
[Reasoning Steps]: Your reasoning steps here.
```
Notice that:
- Remember to wrap the question and answer with triple backticks.
- Don't include the answer in the question or in the reasoning steps.
- Your question should be as objective as possible.
    - Try using the numerical values in the content to ask questions, such as comparing, calculating differences, etc.
- Your answer should be concise and clear.
- Your reasoning steps should serve as a hint to the question. Focus on the action that the participant should do to find the answer, e.g. "Locate the section that describes ...", "Find the name of the model in the section.", but avoid including specific contexts, numbers, titles or captions.
{hint}

Let's think step-by-step, and then provide the final question and answer.
"""

DESCRIPTION_PROMPT = {
"section": "the content of the section in MARKDOWN format",
"page": "the content of the pagg",
"table": "the content of the table in HTML format and the caption of the table",
"image": "the content of the image in base64 format and the caption of the image",
"formula": "the content of the formula in MARKDOWN format"
}

HINT_PROMPT = {
"section": "",
"page": "",
"table": """- Try not to include the word `table` in your question.""",
"image": """- Try to indicate the figure in the question by providing indexes, e.g. Figure 2.""",
"formula": """- Try to indicate the formula in the question by providing indexes, e.g. formula (2)."""
}

EXPLORE_PROMPT = {key: _EXPLORE_PROMPT.format(description=DESCRIPTION_PROMPT[key], hint=HINT_PROMPT[key]) for key in DESCRIPTION_PROMPT}

CONTEXT_PROMPT = {
"section": """The content of the section is as follows:
```markdown
{content}
```""",
"page": """The content of the page is as follows:
```txt
{content}
```""",
"table": """The caption of the table is as follows:
```txt
{caption}
```
The content of the table is as follows:
```html
{content}
```""",
"image": """The caption of the image is as follows:
```txt
{caption}
```""",
"formula": """The formula ({index}) is as follows:
```markdown
{formula}
```"""
}

IMAGE_PROMPT = "The content of the image in base64 format is shown below:"
#coding=utf8

_EXPLORE_PROMPT = """You are an intelligent annotation system who is expert in posing questions. 

{description} Your output should be in the following format:
[Thought]: Your thought process.
```txt
[Question]: Your question here.
[Answer]: Your answer here.
[Reasoning Steps]: Your reasoning steps here.
```
Notice that:
- Remember to wrap your output (except [Thought]) with triple backticks.
- Don't include the answer in the question or in the reasoning steps.
- Your question should be as objective as possible.
    - Try using the numerical values in the content to ask questions, such as comparing, calculating differences, etc.
- Your answer should be concise and clear.
- Your reasoning steps should serve as a hint to the question. Focus on the action that the participant should do to find the answer, e.g. "Locate the section that describes ...", "Find the name of the model in the section.", but avoid including specific contexts, numbers, titles or captions.
{hint}

Let's think step-by-step, and then provide the final question and answer.
"""

DESCRIPTION_PROMPT = {
    "single": {
        "section": "You will be given an AI research paper, and your task is to generate a question based on the content of the section in MARKDOWN format.",
        "page": "You will be given an AI research paper, and your task is to generate a question based on the content of the page.",
        "table": "You will be given an AI research paper, and your task is to generate a question based on the content of the table in HTML format and the caption of the table.",
        "image": "You will be given an AI research paper, and your task is to generate a question based on the content of the image in base64 format and the caption of the image.",
        "formula": "You will be given an AI research paper, and your task is to generate a question based on the content of the formula in MARKDOWN format.",
        "sec_sub": "You will be given an AI research paper, and your task is to generate a question based on the content of the section in MARKDOWN format.",
        "sec_sec": "You will be given an AI research paper, and your task is to generate a question based on the content of the first section and the second section in MARKDOWN format."
    },
    "multiple": {
        "section": "You will be given multiple AI research papers, and your task is to generate a question based on the contents of the sections in MARKDOWN format.",
        "page": "You will be given multiple AI research papers, and your task is to generate a question based on the contents of the pages.",
        "table": "You will be given multiple AI research papers, and your task is to generate a question based on the contents of the tables in HTML format and the captions of the tables.",
        "image": "You will be given multiple AI research papers, and your task is to generate a question based on the contents of the images in base64 format and the captions of the images."
    }
}
DESCRIPTION_PROMPT["comprehensive"] = DESCRIPTION_PROMPT["single"]

HINT_PROMPT = {
    "single": {
        "section": "",
        "page": "",
        "table": """- Try not to include the word `table` in your question.""",
        "image": """- Try to indicate the figure in the question by providing indexes, e.g. Figure 2.""",
        "formula": """- Try to indicate the formula in the question by providing indexes, e.g. formula (2).""",
        "sec_sub": """- Try to pose a sub-question with the text of the section, then pose another sub-question with the text of the subsection.
        - Better make the second sub-question relyng on the first.
        - Note that when you output the question, you should combine the two sub-questions into one question.
        - If there are no subsection, return \"No Subsection.\"""",
        "sec_sec": """- Try to pose a sub-question with the text of the first section, then pose another sub-question with the text of the second section.
        - Better make the second question relyng on the first.
        - Note that when you output the question, you should combine the two sub-questions into one question."""
    },
    "multiple": {
        "section": """- Try to use all the sections to generate the question.""",
        "page": """- Try to use all the pages to generate the question.""",
        "table": """- Try to use all the tables to generate the question.""",
        "image": """- Try to use all the images to generate the question.""",
    },
    "comprehensive": {}
}
for key in HINT_PROMPT["single"]:
    HINT_PROMPT["comprehensive"][key] = HINT_PROMPT["single"][key] + """\n- Try to add qualifiers to make sure the respondents can directly locate the paper, but avoid directly providing the title. e.g. \"In the paper that introduces ReACT ...\", \"In transformer, what's ...\"."""

EXPLORE_PROMPT = {
    category: {
        key: _EXPLORE_PROMPT.format(
            description=DESCRIPTION_PROMPT[category][key], 
            hint=HINT_PROMPT[category][key]
        ) 
        for key in DESCRIPTION_PROMPT[category]
    }
    for category in DESCRIPTION_PROMPT
}

CONTEXT_PROMPT = {
    "single": {
        "section": """The content of the section is as follows:\n```markdown\n{content}\n```""",
        "page": """The content of the page is as follows:\n```txt\n{content}\n```""",
        "table": """The caption of the table is as follows:\n```txt\n{caption}\n```
        The content of the table is as follows:\n```html\n{content}\n```""",
        "image": """The caption of the image is as follows:\n```txt\n{caption}\n```""",
        "formula": """The formula ({index}) is as follows:\n```markdown\n{formula}\n```""",
        "sec_sub": """The content of the section is as follows:\n```markdown\n{content}\n```""",
        "sec_sec": """The content of the first section is as follows:\n```markdown\n{content0}\n```
        The content of the second section is as follows:\n```markdown\n{content1}\n```"""
    },
    "multiple": {
        "section": """The section content of the paper {index} is as follows:\n```markdown\n{content}\n```""",
        "page": """The page content of the paper {index} is as follows:\n```txt\n{content}\n```""",
        "table": """Paper {index}:
        The caption of the table is as follows:\n```txt\n{caption}\n```
        The content of the table is as follows:\n```html\n{content}\n```""",
        "image": """Paper {index}:
        The caption of the image is as follows:\n```txt\n{caption}\n```""",
    }
}
CONTEXT_PROMPT["comprehensive"] = CONTEXT_PROMPT["single"]

IMAGE_PROMPT = {
    "single": "The image in base64 format is shown below:",
    "multiple": "The image of paper {index} in base64 format is shown below:",
    "comprehensive": "The image in base64 format is shown below:"
}
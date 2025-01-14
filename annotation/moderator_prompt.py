#coding=utf8

MODERATE_PROMPT = """You are an intelligent annotation system who is expert in reviewing questions.

You will be given a question and an answer. You should adjust the question and the answer, adapting them to the evaluator's requirements. The descriptions, parameters and use cases of the evaluators are provided below:

------------------------------------------------------------

{evaluator}

------------------------------------------------------------

Your output should be in the following format:
```txt
[question]: Modified question.
[evaluator]: The evaluator you choose. You should present it in JSON format, as given in the use cases.
[answer_format]: The format that the answer should follow in order to pass the evaluator. It will be provided to the respondent along with the question. e.g. "Your answer should be a single python list containing two strings, the first element of the list is the abbreviation of the baseline, the second element of the list is the full name of this baseline, e.g.["MAML","Model-Agnostic Meta-Learning"]." You shouldn't include answers, hints or key points in the answer_format, just focus on the format.
[answer]: A possible answer that can pass the evaluator.
[tag]: A single `subjective` or `objective` without explanation. Whether the evaluator involves LLM. `subjective` if involves LLM, otherwise `objective`.
```

Here're the original question and answer:
```txt
[question]: {question}
[answer]: {answer}
```

Let's think step-by-step, and then provide the final arguments.
"""

EVALUATOR_PROMPT = """## {function}

### Description
{description}

### Parameters
{parameters}

### Use Case(s)

{use_cases}

"""

USECASE_PROMPT = """#### Use Case {index}
{example}
{explanation}
"""
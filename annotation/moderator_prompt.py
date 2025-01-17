#coding=utf8

MODERATE_PROMPT = """You are an intelligent annotation system who is expert in reviewing questions.

You will be given a question and an answer. You should adjust the question and the answer, adapting them to the evaluator's requirements. The descriptions, parameters and use cases of the evaluators are provided below:

------------------------------------------------------------

{evaluator}

Note that:
- If you want the predicted answer list to be exactly same with the gold answer list, use `eval_structured_object_exact_match`, don't use `eval_element_list_included`.
- If your evaluation involves list matching, and the order doesn't matter, set `ignore_order` to `true`. If the order matters, set `ignore_order` to `false`.
- If you are sure that the answer is unique, there aren't other equivalent answers, and any rephrase will change the semantic meaning of the answer, you can use `eval_string_exact_match`. Otherwise, you should use `eval_reference_answer_with_llm`. Generally, we recommend using `eval_reference_answer_with_llm` for subjective questions, and `eval_string_exact_match` for single-word answers.

------------------------------------------------------------

Your output should be in the following format:
[thought]: Your thought process.
```txt
[question]: Modified question.
[evaluator]: The evaluator you choose.
[answer_format]: The format that the respondent should follow in order to pass the evaluator. e.g. \"Your answer should be a single python list containing two strings, the first element of the list is the abbreviation of the baseline, the second element of the list is the full name of this baseline, e.g.[\"abbr\",\"full\"].\".
[answer]: Modified answer.
[tag]: A single `subjective` or `objective` without explanation. Whether the evaluator involves LLM. `subjective` if it involves LLM, otherwise `objective`.
```

Note that:
- Remember to wrap your output (except thought) with triple backticks.
- DON'T INCLUDE ANSWERS, HINTS OR KEY POINTS IN [question] OR [answer_format] IN ANY FORM, ESPECIALLY WHEN YOU TRY TO ILLUSTRATE [answer_format] BY GIVING EXAMPLES.
- [answer_format] will be provided to the respondent along with the [question]. [question] and [answer_format] together form the who question that will be presented to the respondent. [question] focuses on the question itself, [answer_format] focuses on the format of the answer.
- You should present [evaluator] in JSON format, as given in the use cases. And your [answer] should be able to pass the evaluator.
- You can modify the question and answer based on the evaluator's requirements, but don't change the original meaning of the question and answer.
- When the question involves percentage, and the percentage is an exact value, not an approximate value, try to use `eval_float_exact_match` or `eval_int_exact_match`, while indicating the decimal places in [answer_format].

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
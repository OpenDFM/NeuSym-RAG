#coding=utf8
import os, sys, json
from abc import ABC, abstractmethod
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from annotation.moderator_prompt import MODERATOR_PROMPT, EVALUATOR_PROMPT, USECASE_PROMPT
from utils.functions.common_functions import call_llm_with_pattern, call_llm
EVALUATIONS_FILE = os.path.join('evaluation', 'evaluations.json')
EVALUATIONS = json.load(open(EVALUATIONS_FILE, 'r'))

evaluators_prompt = ""
for eval_func in EVALUATIONS:
    evaluation = EVALUATIONS[eval_func]
    evaluators_prompt += EVALUATOR_PROMPT.format(
        function = eval_func,
        description = evaluation['description'],
        parameters = evaluation['parameters'],
        use_cases = "\n".join([
            USECASE_PROMPT.format(
                index = idx,
                example = usecase['example'],
                explanation = usecase['explanation']
            ) for idx, usecase in enumerate(evaluation['use_cases'], start=1)
        ])
    )

# moderator_prompt = MODERATOR_PROMPT.format(
#     evaluator = evaluators_prompt,
#     question = "Which meta learning-based baseline is used in the paper named \"Can We Continually Edit Language Models?On the Knowledge Attenuation in Sequential Model Editing\"? What's the full name of this baseline according to the paper where it's proposed?",
#     answer = "MEND is used in the paper, and its full name is Model Editor Networks with Gradient Decomposition."
# )
moderator_prompt = MODERATOR_PROMPT.format(
    evaluator = evaluators_prompt,
    question = "How to initialize $h_{i,l-1}^S$ in Equation (11)?",
    answer = "To initialize $h_{i,0}^S$ in Equation (11), we start with the output from Equation (8): $h_i^S = \\hat{h}_i + s_i$.\n\nHere, $\\hat{h}_i$ is the output from the Aspect-Aware Attention Module (A3M), and $s_i$ is the sentiment feature obtained by projecting the affective score of word $w_i$ from SenticNet into the same dimensional space as $\\hat{h}_i$. Specifically:\n\n1. For each word $w_i$ in the sentence, obtain its affective score $w_i^S$ from SenticNet.\n2. Project this affective score into the same dimensional space as $\\hat{h}_i$: $s_i = W_S w_i^S + b_S$, where $W_S$ and $b_S$ are learned parameters.\n3. Add the sentiment feature $s_i$ to $\\hat{h}_i$: $h_i^S = \\hat{h}_i + s_i$\n\nThis $h_i^S$ serves as the initial node representation $h_{i,0}^S$ for the Aspect-Guided Graph Convolutional Network (AG-GCN). Therefore, for the first layer $l=0$: $h_{i,0}^S = h_i^S$."
)

print(moderator_prompt)

print(call_llm(moderator_prompt))

class BaseModerator(ABC):
    """ Moderate the question and fill the other parameters.
    1. Reform the question and the answer.
    2. Consider `evaluator`, `answer_format` and `tags`.
    """
    
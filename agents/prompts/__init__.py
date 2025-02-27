#coding=utf8
from agents.prompts.system_prompt import (
    TRIVIAL_SYSTEM_PROMPT,
    CLASSIC_RAG_SYSTEM_PROMPT,
    ITERATIVE_CLASSIC_RAG_SYSTEM_PROMPT,
    ITERATIVE_NEU_RAG_SYSTEM_PROMPT,
    ITERATIVE_SYM_RAG_SYSTEM_PROMPT,
    NEUSYM_RAG_SYSTEM_PROMPT,
    TWO_STAGE_SYM_RAG_SYSTEM_PROMPT,
    TWO_STAGE_NEU_RAG_SYSTEM_PROMPT,
    TWO_STAGE_HYBRID_RAG_SYSTEM_PROMPT
)

from agents.prompts.agent_prompt import (
    CLASSIC_RAG_AGENT_PROMPT,
    ITERATIVE_RAG_AGENT_PROMPT,
    TWO_STAGE_NEU_RAG_AGENT_PROMPT,
    TWO_STAGE_SYM_RAG_AGENT_PROMPT,
    TWO_STAGE_HYBRID_RAG_AGENT_PROMPT
)

from agents.prompts.hint_prompt import (
    ITERATIVE_CLASSIC_RAG_HINT_PROMPT,
    ITERATIVE_NEU_RAG_HINT_PROMPT,
    ITERATIVE_SYM_RAG_HINT_PROMPT,
    NEUSYM_RAG_HINT_PROMPT
)
from agents.prompts.schema_prompt import convert_database_schema_to_prompt, convert_vectorstore_schema_to_prompt
from agents.prompts.task_prompt import formulate_input


SYSTEM_PROMPTS = {
    'trivial_question_only': TRIVIAL_SYSTEM_PROMPT,
    'trivial_title_with_abstract': CLASSIC_RAG_SYSTEM_PROMPT,
    'trivial_full_text_with_cutoff': CLASSIC_RAG_SYSTEM_PROMPT,
    'classic_rag': CLASSIC_RAG_SYSTEM_PROMPT,
    'iterative_classic_rag': ITERATIVE_CLASSIC_RAG_SYSTEM_PROMPT,
    'two_stage_neu_rag': TWO_STAGE_NEU_RAG_SYSTEM_PROMPT,
    'iterative_neu_rag': ITERATIVE_NEU_RAG_SYSTEM_PROMPT,
    'two_stage_sym_rag': TWO_STAGE_SYM_RAG_SYSTEM_PROMPT,
    'iterative_sym_rag': ITERATIVE_SYM_RAG_SYSTEM_PROMPT,
    # 'two_stage_graph_rag': TWO_STAGE_GRAPH_RAG_SYSTEM_PROMPT,
    # 'iterative_graph_rag': ITERATIVE_GRAPH_RAG_SYSTEM_PROMPT,
    'two_stage_hybrid_rag': TWO_STAGE_HYBRID_RAG_SYSTEM_PROMPT,
    'neusym_rag': NEUSYM_RAG_SYSTEM_PROMPT
}


HINT_PROMPTS = {
    "iterative_classic_rag": ITERATIVE_CLASSIC_RAG_HINT_PROMPT,
    "iterative_neu_rag": ITERATIVE_NEU_RAG_HINT_PROMPT,
    "iterative_sym_rag": ITERATIVE_SYM_RAG_HINT_PROMPT,
    # "iterative_graph_rag": ITERATIVE_GRAPH_RAG_HINT_PROMPT,
    "neusym_rag": NEUSYM_RAG_HINT_PROMPT
}

AGENT_PROMPTS = {
    'trivial_question_only': CLASSIC_RAG_AGENT_PROMPT,
    'trivial_title_with_abstract': CLASSIC_RAG_AGENT_PROMPT,
    'trivial_full_text_with_cutoff': CLASSIC_RAG_AGENT_PROMPT,
    'classic_rag': CLASSIC_RAG_AGENT_PROMPT,
    'iterative_classic_rag': ITERATIVE_RAG_AGENT_PROMPT,
    'two_stage_neu_rag': TWO_STAGE_NEU_RAG_AGENT_PROMPT,
    'iterative_neu_rag': ITERATIVE_RAG_AGENT_PROMPT,
    'two_stage_sym_rag': TWO_STAGE_SYM_RAG_AGENT_PROMPT,
    'iterative_sym_rag': ITERATIVE_RAG_AGENT_PROMPT,
    # 'two_stage_graph_rag': TWO_STAGE_GRAPH_RAG_AGENT_PROMPT,
    # 'iterative_graph_rag': ITERATIVE_RAG_AGENT_PROMPT,
    'two_stage_hybrid_rag': TWO_STAGE_HYBRID_RAG_AGENT_PROMPT,
    'neusym_rag': ITERATIVE_RAG_AGENT_PROMPT
}

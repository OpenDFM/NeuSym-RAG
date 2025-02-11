#coding=utf8
from agents.prompts.system_prompt import TRIVIAL_SYSTEM_PROMPT, CLASSIC_RAG_SYSTEM_PROMPT, TEXT2SQL_SYSTEM_PROMPT, TEXT2VEC_SYSTEM_PROMPT, TWO_STAGE_TEXT2SQL_SYSTEM_PROMPT, TWO_STAGE_TEXT2VEC_SYSTEM_PROMPT, HYBRID_RAG_SYSTEM_PROMPT, TWO_STAGE_HYBRID_SYSTEM_PROMPT
from agents.prompts.agent_prompt import TRIVIAL_PROMPT, CLASSIC_RAG_PROMPT, REACT_PROMPT, TWO_STAGE_TEXT2SQL_PROMPT, TWO_STAGE_TEXT2VEC_PROMPT, TWO_STAGE_HYBRID_PROMPT
from agents.prompts.hint_prompt import TEXT2SQL_HINT_PROMPT, TEXT2VEC_HINT_PROMPT, HYBRDI_RAG_HINT_PROMPT
from agents.prompts.schema_prompt import convert_database_schema_to_prompt, convert_vectorstore_schema_to_prompt
from agents.prompts.task_prompt import formulate_input

SYSTEM_PROMPTS = {
    'trivial': TRIVIAL_SYSTEM_PROMPT,
    'classic_rag': CLASSIC_RAG_SYSTEM_PROMPT,
    'text2sql': TEXT2SQL_SYSTEM_PROMPT,
    'text2vec': TEXT2VEC_SYSTEM_PROMPT,
    'hybrid_rag': HYBRID_RAG_SYSTEM_PROMPT,
    'two_stage_text2sql': TWO_STAGE_TEXT2SQL_SYSTEM_PROMPT,
    'two_stage_text2vec': TWO_STAGE_TEXT2VEC_SYSTEM_PROMPT,
    'two_stage_hybrid': TWO_STAGE_HYBRID_SYSTEM_PROMPT
}

HINT_PROMPTS = {
    'text2sql': TEXT2SQL_HINT_PROMPT,
    'text2vec': TEXT2VEC_HINT_PROMPT,
    'hybrid_rag': HYBRDI_RAG_HINT_PROMPT
}

AGENT_PROMPTS = {
    'trivial': TRIVIAL_PROMPT,
    'classic_rag': CLASSIC_RAG_PROMPT,
    'react': REACT_PROMPT,
    'two_stage_text2sql': TWO_STAGE_TEXT2SQL_PROMPT,
    'two_stage_text2vec': TWO_STAGE_TEXT2VEC_PROMPT,
    'two_stage_hybrid': TWO_STAGE_HYBRID_PROMPT
}

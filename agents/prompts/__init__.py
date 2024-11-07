#coding=utf8
from agents.prompts.system_prompt import CLASSIC_RAG_SYSTEM_PROMPT, TEXT2SQL_SYSTEM_PROMPT, TEXT2VEC_SYSTEM_PROMPT, TWO_STAGE_TEXT2SQL_SYSTEM_PROMPT, TWO_STAGE_TEXT2VEC_SYSTEM_PROMPT
from agents.prompts.agent_prompt import CLASSIC_RAG_PROMPT, REACT_PROMPT, TWO_STAGE_TEXT2SQL_PROMPT, TWO_STAGE_TEXT2VEC_PROMPT
from agents.prompts.schema_prompt import convert_database_schema_to_prompt, convert_vectorstore_schema_to_prompt
from agents.prompts.task_prompt import formulate_input

SYSTEM_PROMPTS = {
    'classic_rag': CLASSIC_RAG_SYSTEM_PROMPT,
    'text2sql': TEXT2SQL_SYSTEM_PROMPT,
    'text2vec': TEXT2VEC_SYSTEM_PROMPT,
    'two_stage_text2sql': TWO_STAGE_TEXT2SQL_SYSTEM_PROMPT,
    'two_stage_text2vec': TWO_STAGE_TEXT2VEC_SYSTEM_PROMPT
}

AGENT_PROMPTS = {
    'classic_rag': CLASSIC_RAG_PROMPT,
    'react': REACT_PROMPT,
    'two_stage_text2sql': TWO_STAGE_TEXT2SQL_PROMPT,
    'two_stage_text2vec': TWO_STAGE_TEXT2VEC_PROMPT
}

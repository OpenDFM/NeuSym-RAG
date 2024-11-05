#coding=utf8
from agents.prompts.system_prompt import TEXT2SQL_SYSTEM_PROMPT, TWO_STAGE_TEXT2SQL_SYSTEM_PROMPT
from agents.prompts.agent_prompt import REACT_PROMPT, TWO_STAGE_TEXT2SQL_PROMPT
from agents.prompts.schema_prompt import convert_database_schema_to_prompt
from agents.prompts.task_prompt import formulate_input

SYSTEM_PROMPTS = {
    'text2sql': TEXT2SQL_SYSTEM_PROMPT,
    'two_stage_text2sql': TWO_STAGE_TEXT2SQL_SYSTEM_PROMPT
}

AGENT_PROMPTS = {
    'react': REACT_PROMPT,
    'two_stage_text2sql': TWO_STAGE_TEXT2SQL_PROMPT
}

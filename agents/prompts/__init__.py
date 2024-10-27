#coding=utf8
from agents.prompts.system_prompt import TEXT2SQL_SYSTEM_PROMPT, TWOSTAGE_TEXT2SQL_SYSTEM_PROMPT
from agents.prompts.agent_prompt import REACT_PROMPT, TWOSTAGE_PROMPT
from agents.prompts.database_schema_prompt import convert_database_schema_to_prompt
from agents.prompts.task_prompt import formulate_input

SYSTEM_PROMPTS = {
    'text2sql': TEXT2SQL_SYSTEM_PROMPT,
    'twostage_text2sql': TWOSTAGE_TEXT2SQL_SYSTEM_PROMPT
}

AGENT_PROMPTS = {
    'react': REACT_PROMPT,
    'twostage': TWOSTAGE_PROMPT
}
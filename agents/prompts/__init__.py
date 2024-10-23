#coding=utf8
from agents.prompts.system_prompt import TEXT2SQL_SYSTEM_PROMPT
from agents.prompts.agent_prompt import REACT_PROMPT
from agents.prompts.database_schema_prompt import convert_database_schema_to_prompt


SYSTEM_PROMPTS = {
    'text2sql': TEXT2SQL_SYSTEM_PROMPT
}

AGENT_PROMPTS = {
    'react': REACT_PROMPT
}
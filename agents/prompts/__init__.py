#coding=utf8
from agents.prompts.system_prompt import TEXT2SQL_BASELINE_SYSTEM_PROMPT
from agents.prompts.action_and_observation_prompt import TEXT2SQL_ACTION_AND_OBSERVATION_PROMPT
from agents.prompts.database_schema_prompt import convert_database_schema_to_prompt
from agents.prompts.agent_framework_prompt import TEXT2SQL_REACT_PROMPT


SYSTEM_PROMPTS = {
    'text2sql+react': TEXT2SQL_REACT_PROMPT
}
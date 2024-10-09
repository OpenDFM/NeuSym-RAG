#coding-utf8
import re
from typing import List, Tuple, Dict, Any
from agents.parsers.base_output_parser import BaseOutputParser


class Text2SQLReActOutputParser(BaseOutputParser):
    """ Refer to TEXT2SQL_ACTION_AND_OBSERVATION_PROMPT in prompts/action_and_observation_prompt.py
    """
    def parse(self, output: str) -> Dict[str, Any]:
        """ Parse the output of the text2sql model.
        @param:
            output: str, the output of the text2sql model
        @return:
            parsed_output: Dict[str, Ant], parse the raw LLM response into structured dict containing
                {
                    "action_type": "GenerateSQL", // or "GenerateAnswer",
                    "action": "concrete action content or parameters",
                    "thought": "thought process" // optional
                }
        """
        parsed_output = {'thought': '', 'action_type': '', 'action': output.strip()}
        thought_pattern = r'Thought:\s*(.*?)\s*Action:'
        thought = re.search(thought_pattern, output, flags=re.DOTALL)
        if thought:
            parsed_output['thought'] = thought.group(1).strip()

        generate_sql_pattern = r'GenerateSQL:\s*```(sql)?\s*(.*?)\s*```'
        match = re.search(generate_sql_pattern, output, flags=re.DOTALL)
        if match:
            parsed_output['action_type'] = 'GenerateSQL'
            parsed_output['action'] = match.group(2).strip()
            return parsed_output

        generate_answer_pattern = r'GenerateAnswer:\s*```(txt)?\s*(.*?)\s*```'
        match = re.search(generate_answer_pattern, output, flags=re.DOTALL)
        if match:
            parsed_output['action_type'] = 'GenerateAnswer'
            parsed_output['action'] = match.group(2).strip()
            return parsed_output
        
        return parsed_output
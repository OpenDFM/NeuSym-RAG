#coding=utf8
import re, json, os, ast
import xmltodict, yaml
import gymnasium as gym
from bs4 import BeautifulSoup
from dataclasses import dataclass, field, fields
from typing import Optional, List, Tuple, Dict, Union, Any
from abc import ABC, abstractmethod
from agents.envs.actions.observation import Observation


ACTIONS_FILE = os.path.join(os.path.dirname(__file__), 'actions.json')
ACTIONS = json.load(open(ACTIONS_FILE, 'r'))


class ParseActionError(ValueError):
    pass

class ParseParametersError(ValueError):
    pass

class MismatchedActionError(ValueError):
    pass


ACTION_FORMATS = ['markdown', 'json', 'xml', 'yaml'] # allowable action formats


def extract_inner_text(text: str, prefix: str = '{', suffix: str = '}') -> str:
    """ Extract the JSON or XML text from the raw LLM response.
    """
    if prefix not in text or suffix not in text:
        return text
    start = text.index(prefix) if prefix != '' else 0
    end = text.rindex(suffix) if suffix != '' else len(text)
    inner_text = text[start: end + len(suffix)]
    return inner_text


def soup_to_dict(element):
    """ Recursively convert a BeautifulSoup element to a JSON dictionary.
    """
    if not element.find_all():
        return element.text.strip()
    return {child.name: soup_to_dict(child) for child in element.find_all(recursive=False)}


@dataclass
class Action(ABC):

    thought: str = field(default='', repr=False) # reasoning process for popular agent frameworks like ReAct
    observation_format_kwargs: Dict[str, Any] = field(default_factory=dict, repr=False) # default keyword arguments for observation formatting

    @property
    def done(self) -> bool:
        return False


    @abstractmethod
    def execute(self, env: gym.Env, **format_kwargs) -> Observation:
        """ Execute the action in the environment and return the Observation object.
        """
        pass


    @classmethod
    def specification(cls, action_format: str = 'json') -> str:
        """ Return a human-readable specification of the action according to the argument `action_format`.
        This specification is usually inserted into the action space of the system prompt. The checklist of all actions is defined in file `actions.json`, each as a json dict like the following example: (will be automatically converted into the specified `action_format`)
        {
            "action_type": "GenerateAnswer",
            "description": "Generate the final answer based on the retrieved context and interaction log.",
            "observation": "No observation, this is the terminal action.",
            "parameters": { // each field is defined as a dict of these keys: `type`, `required`, `default`, `description`, where `default` is only needed for fields that `required=false`
                "answer": {
                    "type": "Any",
                    "required": true,
                    "description": "The final answer to the user question. Please adhere to the answer format for the current question."
                }
            },
            "use_case": [ // each use case as a dict of two keys, where `example` gives the value for each parameter, and `explanation` explains the use case
                {"example": {"answer": 42}, "explanation": "The final answer is 42."},
                {"example": {"answer": ["Results", "Discussion"]}, "explanation": "The final answer is a list of strings: ['Results', 'Discussion']."}
            ]
        }
        """
        action_type = cls.__name__
        if action_type not in ACTIONS:
            raise ValueError(f"Action type `{action_type}` not found in file {ACTIONS_FILE}.")

        action_spec = ACTIONS[action_type]
        action_type = action_spec['action_type']
        description = action_spec['description']
        observation = action_spec['observation']
        
        parameters = action_spec['parameters']
        if action_format == 'markdown':
            param_names = [field_name for field_name, field_spec in parameters.items() if field_spec.get("required", True)] + [field_name for field_name, field_spec in parameters.items() if not field_spec.get("required", True)]
            params = ', '.join([
                f'{field_name}: {parameters[field_name]["type"]}'
                if parameters[field_name].get("required", True)
                else f'{field_name}: {parameters[field_name]["type"]} = {repr(parameters[field_name]["default"])}'
                for field_name in param_names
            ])
            comments = [f'- {field_name}: {parameters[field_name]["type"]}, {"required" if parameters[field_name].get("required", True) else "optional, default to " + repr(parameters[field_name]["default"])}. {parameters[field_name]["description"]}' for field_name in param_names]
            syntax = f"{action_type}({params})\n" + '\n'.join(comments)
        elif action_format == 'json':
            syntax = json.dumps({'action_type': action_type, 'parameters': parameters}, indent=4, ensure_ascii=False)
        elif action_format == 'xml':
            syntax = xmltodict.unparse({'action': {'action_type': action_type, 'parameters': parameters}}, pretty=True, indent=4, encoding='utf-8')
            if syntax.startswith("<?xml"): # ignore the first line of <?xml>
                syntax = syntax.split("?>", 1)[1].strip()
        elif action_format == 'yaml':
            syntax = yaml.dump({'action_type': action_type, 'parameters': parameters}, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=4)
        else:
            raise ValueError(f"Action format {action_format} not supported yet.")

        use_cases = action_spec['use_cases']
        use_cases_prompt = []
        for idx, case in enumerate(use_cases):
            if action_format == 'markdown':
                action_str = f'{action_type}(' + ', '.join([f'{field_name}={repr(case["example"][field_name])}' for field_name in param_names if field_name in case["example"]]) + ')'
            elif action_format == 'json':
                action_str = json.dumps({'action_type': action_type, 'parameters': case['example']}, ensure_ascii=False)
            elif action_format == 'xml':
                action_str = xmltodict.unparse({'action': {'action_type': action_type, 'parameters': case['example']}}, encoding='utf-8')
                if action_str.startswith("<?xml"):
                    action_str = action_str.split("?>", 1)[1].strip()
            elif action_format == 'yaml':
                action_str = '\n' + yaml.dump({'action_type': action_type, 'parameters': case['example']}, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=4) + '\n'
            else:
                raise ValueError(f"Action format {action_format} not supported yet.")
            action = f"[Action]: {action_str}"
            explanation = f'[Explanation]: {case["explanation"]}'
            use_cases_prompt.append(f"\n#### Case {idx + 1}\n{action}\n{explanation}\n")
        use_cases_prompt = '\n'.join(use_cases_prompt)

        action_prompt = f"""
### Action Type
{action_type}

### Description
{description}

### Observation
{observation}

### Syntax and Parameters ({action_format.upper()} Format)
{syntax}

### Use Cases
{use_cases_prompt}
"""
        return action_prompt


    @classmethod
    def get_action_space_prompt(cls, action_types: List[type], action_format: str = 'json') -> str:
        """ Return the entire action space prompt for all given action types (using function `_specification`) based on the `action_format`.
        """
        assert action_format in ACTION_FORMATS, f"Action format {action_format} not supported."
        action_names = [action_cls.__name__ for action_cls in action_types]
        action_space_prompt = f"## Action and Observation Space\nAll allowable action types include {str(action_names)}. Here is the detailed specification in {action_format.upper()} format:\n"
        actions = []
        for action_cls in action_types:
            actions.append(action_cls.specification(action_format))
        return action_space_prompt + '\n----\n'.join(actions)


    @classmethod
    def parse_action(cls, text: str, action_types: List[type], action_format: str = 'json', agent_method: str = 'react') -> Tuple[bool, 'Action']:
        """ Parse the raw LLM response text into one concrete Action object based on the allowable action types and the specified action `format`.
        @args:
            text: str, the raw LLM response text
            action_types: List[type], a list of allowable action types, depending on the environment
            action_format: str, the format of the action text, chosen from ['markdown', 'json', 'xml', 'yaml']
            agent_method: str, the agent method for the response, used to extract the parsable action_text from raw LLM response text, chosen from ['react', 'code_block'], default to 'react'
                - react: each action should be wrapped in the framework below (`Thought` is optional)
                    [Thought]: ...
                    [Action]: ...
                    [Observation]: ...
                - code_block: each action should be wrapped in the 3 backticks
        @return:
            flag: bool, whether the action is successfully parsed
            action_obj: Action, the parsed action object
        """
        assert action_format in ACTION_FORMATS, f"Action format `{action_format}` is not supported."

        # extract the real action_text from raw LLM response, maybe dependent on agent frameworks
        # currently only support react and code_block styles
        if agent_method not in ['react', 'code_block']: agent_method = 'react'
        if agent_method == 'react':
            thought_pattern = r"\[Thought\]:\s*(.*?)\s*\[Action\]:"
            matched_thought = re.search(thought_pattern, text, re.DOTALL)
            thought = matched_thought.group(1) if matched_thought else None
            action_pattern = r"\[Action\]:\s*(.*?)\s*(\[Observation\]:|$)"
            matched_action = re.search(action_pattern, text, re.DOTALL)
            action_text = matched_action.group(1).strip() if matched_action else text.strip()
        else: # agent_method == 'code_block':
            thought = None
            matching_list = re.findall(r"```(\S*)\s*(.*?)\s*```", text.strip(), flags=re.DOTALL)
            action_text = matching_list[-1][1].strip() if len(matching_list) > 0 else text.strip()

        from .error_action import ErrorAction
        action_names = [action_cls.__name__ for action_cls in action_types]
        for action_cls in action_types:
            try:
                action_obj = action_cls._parse(action_text, action_format)
                if thought is not None:
                    action_obj.thought = thought # add thought to the action object
                return True, action_obj
            except ParseActionError as e: # failed to parse the action structure, e.g., json, xml, etc.
                return False, ErrorAction(response=text, error=str(e))
            except ParseParametersError as e: # failed to parse the action parameters for a specific action
                return False, ErrorAction(response=text, error=str(e))
            # except MismatchedActionError:
                # continue # try next action type
            except Exception as e:
                continue # try next action type
        return False, ErrorAction(response=text, error=f"Failed to parse a valid action from the response. Please check the specification for these actions {str(action_names)}.")


    @classmethod
    def _parse(cls, action_text: str, action_format: str = 'json') -> 'Action':
        """ Parse the action text into the concrete Action object based on the specified `action_format`.
        """
        class_name = cls.__name__
        if action_format == 'markdown':
            action_type = re.search(r"(.*?)\(", action_text.strip())
            if action_type is None or action_type.group(1).strip() != class_name:
                raise MismatchedActionError(f"The current response does not match {class_name} action.")
            try:
                tree = ast.parse(action_text, mode='eval')
                positional_args, keyword_args = [], {}

                assert isinstance(tree.body, ast.Call)
                for arg in tree.body.args:
                    positional_args.append(ast.literal_eval(arg))
                for kwarg in tree.body.keywords:
                    keyword_args[kwarg.arg] = ast.literal_eval(kwarg.value)
                return cls(*positional_args, **keyword_args)
            except Exception as e:
                raise ParseParametersError(f"Failed to parse the parameters for action {class_name} from the response. {str(e)}.")

        elif action_format == 'json':
            action_text = extract_inner_text(action_text, prefix='{', suffix='}')
            try:
                action_dict: dict = json.loads(action_text.strip())
            except Exception as e:
                raise ParseActionError(f"Failed to parse a valid JSON dict from the response. {str(e)}.")

            if action_dict.get('action_type', '') != class_name:
                raise MismatchedActionError(f"The current response does not match {class_name} action.")
            try:
                return cls(**action_dict['parameters'])
            except Exception as e:
                raise ParseParametersError(f"Failed to parse the parameters for action {class_name} from the response. {str(e)}.")

        elif action_format == 'xml':
            action_text = extract_inner_text(action_text, prefix='<action>', suffix='</action>')
            try:
                # Attention: each value is parsed as a string
                soup = BeautifulSoup(action_text, "xml")
                # Convert XML to dictionary
                action_dict = {soup.find().name: soup_to_dict(soup.find())}['action']
                # [Deprecated]: xmltodict.parse often with bugs
                # action_dict = xmltodict.parse(action_text.strip())['action']
            except Exception as e:
                raise ParseActionError(f"Failed to parse a valid XML object from the response. {str(e)}.")

            if action_dict.get('action_type', '') != class_name:
                raise MismatchedActionError(f"The current response does not match {class_name} action.")
            try:
                return cls(**action_dict['parameters'])
            except Exception as e:
                raise ParseParametersError(f"Failed to parse the parameters for action {class_name} from the response. {str(e)}.")
        
        elif action_format == 'yaml':
            action_text = extract_inner_text(action_text, prefix='action_type:', suffix='')
            try:
                if 'action_type:' not in action_text:
                    action_text = 'action_type: ' + action_text
                action_dict: dict = yaml.safe_load(action_text.strip())
            except Exception as e:
                raise ParseActionError(f"Failed to parse a valid YAML object from the response. {str(e)}.")
            if action_dict.get('action_type', '') != class_name:
                raise MismatchedActionError(f"The current response does not match {class_name} action.")
            try:
                return cls(**action_dict['parameters'])
            except Exception as e:
                raise ParseParametersError(f"Failed to parse the parameters for action {class_name} from the response. {str(e)}.")
        else:
            raise ValueError(f"Action format {action_format} not supported for {class_name} action yet.")


    def convert_to_message(self, action_format: str = 'json') -> Dict[str, str]:
        """ Convert the Action object into a message according to the specified format.
        This message is used to record the interaction history.
        """
        action_str = ''
        if hasattr(self, 'thought') and self.thought is not None:
            action_str += f'[Thought]: {self.thought}\n'

        action_str += '[Action]: '
        cls_type, cls_name = self.__class__, self.__class__.__name__
        if action_format == 'markdown':
            action_str += repr(self) # directly use the repr() function for dataclass
        elif action_format == 'json':
            action_str += json.dumps({
                "action_type": cls_name,
                "parameters": {
                    field.name: getattr(self, field.name)
                    for field in fields(cls_type)
                    if field.repr
                }
            }, indent=4)
        elif action_format == 'xml':
            json_dict = {
                "action": {
                    "action_type": cls_name,
                    "parameters": {
                        field.name: getattr(self, field.name)
                        for field in fields(cls_type)
                        if field.repr
                    }
                }
            }
            action_str = xmltodict.unparse(json_dict, pretty=True, indent=4, encoding='utf-8')
            if action_str.startswith("<?xml"): # ignore the first line of <?xml>
                action_str = action_str.split("?>", 1)[1].strip()
        elif action_format == 'yaml':
            json_dict = {
                "action_type": cls_name,
                "parameters": {
                    field.name: getattr(self, field.name)
                    for field in fields(cls_type)
                    if field.repr
                }
            }
            action_str = '\n' + yaml.dump(json_dict, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=4)
        else:
            raise ValueError(f"Action format {action_format} not supported for {cls_name}.")
        return {'role': 'assistant', 'content': action_str}
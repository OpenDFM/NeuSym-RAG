#coding=utf8
import re, json
import duckdb, xmltodict
import pandas as pd
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Any
from abc import ABC, abstractmethod
from func_timeout import func_set_timeout, FunctionTimedOut


class ParseActionError(ValueError):
    pass

class MismatchedActionError(ValueError):
    pass


@dataclass
class Action(ABC):

    thought: Optional[str] = None # reasoning process for popular agent frameworks like ReAct
    observation_format_kwargs: Dict[str, Any] = {} # default keyword arguments for observation formatting
    observation: Optional[str] = None # observation string for the action
    ACTION_FORMATS = ['markdown', 'json', 'xml'] # allowable action formats

    @property
    def done(self) -> bool:
        return False

    @abstractmethod
    def execute(self, env: gym.Env, **format_kwargs) -> str:
        """ Execute the action in the environment and return the observation string.
        """
        pass

    @abstractmethod
    def serialize(self, action_format: str = 'markdown') -> str:
        """ Serialize the action into a string according to the specified format.
        This serialized action is usually used to log the interaction history message.
        """
        pass

    @abstractmethod
    @classmethod
    def _parse(cls, action_text: str, action_format: str = 'markdown') -> 'Action':
        """ Parse the action text into the concrete Action object based on the specified `action_format`.
        Note that, this is the real parsing function for the current Action class, which is internally invoked by the `parse_action` function.
        """
        pass

    @abstractmethod
    @classmethod
    def _specification(cls, action_format: str = 'markdown') -> str:
        """ Return a human-readable specification of the action according to the specified `action_format`.
        This specification is usually inserted into the action space of the system prompt.
        """
        pass

    @classmethod
    def get_action_space_prompt(cls, action_types: List[type], action_format: str = 'markdown') -> str:
        """ Return the entire action space prompt for all given action types (using function `_specification`) based on the `action_format`.
        """
        assert action_format in Action.ACTION_FORMATS, f"Action format {action_format} not supported."
        action_names = [action_cls.__name__ for action_cls in action_types]
        action_space_prompt = f"## Action and Observation Space\nAll allowable action types include {str(action_names)}. Here is the detailed specification in {action_format.upper()} format:\n\n"
        actions = []
        for action_cls in action_types:
            actions.append(action_cls._specification(action_format))
        return action_space_prompt + '\n----\n'.join(actions)

    @classmethod
    def parse_action(cls, text: str, action_types: List[type], action_format: str = 'markdown') -> Tuple[bool, Union[str, 'Action']]:
        """ Parse the raw LLM response text into one concrete Action object based on the allowable action types and the specified action `format`.
        For a unified perspective, note that each action should be wrapped in (`Thought` is optional)
            [Thought]: ...
            [Action]: ...
            [Observation]: ...
        TODO: maybe this should be combined with the `agent_method='react'` in the future to support more frameworks?
        """
        assert action_format in Action.ACTION_FORMATS, f"Action format {action_format} not supported."
        # extract the real action text from raw LLM response, maybe dependent on agent frameworks
        thought_pattern = r"\[Thought\]:\s*(.*?)\s+\[Action\]:"
        matched_thought = re.search(thought_pattern, text, re.DOTALL)
        thought = matched_thought.group(1) if matched_thought else None
        action_pattern = r"\[Action\]:\s*(.*?)\s+(\[Observation\]:|$)"
        matched_action = re.search(action_pattern, text, re.DOTALL)
        action_text = matched_action.group(1).strip() if matched_action else None
        if action_text is None:
            return False, "[Error]: Failed to parse action from the response."

        for action_cls in action_types:
            try:
                action_obj = action_cls._parse(action_text, action_format)
                action_obj.thought = thought # add thought to the action object
                return True, action_obj
            except ParseActionError:
                return False, f"[Error]: Failed to parse valid parameters for action {action_cls.__name__} from the response, please check the specification."
            # except MismatchedActionError:
                # continue # try next action type
            except Exception as e:
                continue # try next action type
        action_names = [action_cls.__name__ for action_cls in action_types]
        return False, f"[Error]: Failed to parse valid action from the response, please check the specification for these actions {str(action_names)}."


@dataclass
class GenerateSQL(Action):

    sql: str # concrete SQL query, required
    observation_format_kwargs: Dict[str, Any] = {
        "output_format": "markdown", # output format for the SQL execution result, chosen from ['markdown', 'string', 'html'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 50, # maximum rows to display in the output
        "index": False, # whether to include the row index in the output
        "header": True, # whether to include the column names in the output
        "max_timeout": 600 # the maximum timeout for the SQL execution is 10 minutes
    } # keyword arguments for SQL execution formatting

    def execute(self, env: gym.Env, **kwargs) -> str:
        """ Execute the SQL query in the environment and return the formatted observation string.
        For different output formats, see the following references:
            1. pandas.DataFrame.to_markdown(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html
            2. pandas.DataFrame.to_string(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
            3. pandas.DataFrame.to_html(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html
        """
        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists

        @func_set_timeout(0, allowOverride=True)
        def output_formatter(sql: str, format_kwargs: Dict[str, Any], **kwargs) -> str:
            output_format = format_kwargs.pop('output_format', 'markdown')
            assert output_format in ['markdown', 'string', 'html'], "SQL execution output format must be chosen from ['markdown', 'string', 'html']."

            conn: duckdb.DuckDBPyConnection = env.database_conn # get the database connection
            result: pd.DataFrame = conn.execute(sql).fetchdf() # execute the SQL query and fetch the result
            max_rows = format_kwargs.pop('max_rows', 50)
            result = result.head(max_rows)
            suffix = f'\n... # only display {max_rows} rows, more are truncated due to length constraint' if \
                result.shape[0] > max_rows else f'\nIn total, {result.shape[0]} rows are displayed.'
            
            if output_format == 'markdown':
                header = format_kwargs.pop('header', True)
                if not header:
                    print("[Warning]: The column header must be included in the markdown-style SQL output.")
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg = result.to_markdown(**format_kwargs)
            elif output_format == 'string':
                format_kwargs.pop('tablefmt')
                msg = result.to_string(**format_kwargs)
            elif output_format == 'html':
                format_kwargs.pop('tablefmt')
                msg = result.to_html(**format_kwargs)
            else:
                raise ValueError(f"SQL execution output format {output_format} not supported.")
            return msg + suffix

        try:
            max_timeout = output_kwargs.pop('max_timeout', 600)
            msg = output_formatter(self.sql, output_kwargs, forceTimeout=max_timeout)
            return msg
        except FunctionTimedOut as e:
            msg = f"[TimeoutError]: The SQL execution is TIMEOUT given maximum {max_timeout} seconds."
        except Exception as e:
            msg = f"[Error]: {str(e)}"
        return msg


    @classmethod
    def _specification(cls, action_format: str = 'markdown') -> str:
        if action_format == 'markdown': # Markdown format, more similar to raw text, default
            action_format = """GenerateSQL:\n```sql\nconcrete sql query\n```"""
            use_case = """GenerateSQL:\n```sql\nSELECT COUNT(*) FROM table_name;\n```"""
        elif action_format == 'json': # JSON format
            action_format = '{\n    "action_type": "GenerateSQL",\n    "parameters": {\n        "sql": str // concrete sql query, required\n    }\n}'
            example = {
                "action_type": "GenerateSQL",
                "parameters": {
                    "sql": "SELECT COUNT(*) FROM table_name;"
                }
            }
            use_case = json.dumps(example, indent=4)
        elif action_format == 'xml': # XML format
            action_format = """<action><action_type>GenerateSQL</action_type><parameters><sql>concrete SQL query, this is a required tag</sql></parameters></action>"""
            use_case = """<action><action_type>GenerateSQL</action_type><parameters><sql>SELECT COUNT(*) FROM table_name;</sql></parameters></action>"""
        else:
            raise ValueError(f"Action format {action_format} not supported for GenerateSQL.")
        specification = f"""
### Action Type
GenerateSQL

### Action Format
{action_format}

### Description
Generate an SQL query to retrieve the desired information from the database. Please refer to the concrete database schema to produce the valid SQL.

### Observation
The observation space is the execution result of the SQL query. You do not need to worry about the actual execution, we will perform it for you. If the SQL failed to execute, we will return the error message. And extremely long SQL output will be truncated for further actions.

### Use Case
{use_case}
"""
        return specification


    def serialize(self, action_format = 'markdown') -> str:
        """ Serialize the action into a string according to the specified format.
        Attention: please conform to the specification.
        """
        action_str = ''
        if hasattr(self, 'thought') and self.thought is not None:
            action_str += f'[Thought]: {self.thought}\n'
        
        action_str += '[Action]: '
        if action_format == 'markdown':
            action_str += f"GenerateSQL:\n```sql\n{self.sql}\n```"
        elif action_format == 'json':
            action_str += json.dumps({
                "action_type": "GenerateSQL",
                "parameters": {
                    "sql": self.sql
                }
            }, indent=4)
        elif action_format == 'xml':
            action_str += f"<action><action_type>GenerateSQL</action_type><parameters><sql>{self.sql}</sql></parameters></action>"
        else:
            raise ValueError(f"Action format {action_format} not supported for GenerateSQL.")
        return action_str


    @classmethod
    def _parse(cls, action_text: str, action_format: str = 'markdown') -> Action:
        """ Try to parse the Action object from the raw text, if failed, raise an Exception.
        Attention: please conform to the specification.
        """
        if action_format == 'markdown':
            action_type = re.search(r"(.*?):\s*```", action_text.strip())
            if action_type is None or action_type.group(1).strip() != 'GenerateSQL:':
                raise MismatchedActionError("Failed to parse GenerateSQL action from the response.")
            
            sql = re.search(r"GenerateSQL:\s*```(sql)?\s*(.*?)\s*```", action_text.strip(), flags=re.DOTALL)
            if sql is None:
                raise ParseActionError("Failed to parse the SQL query from the response.")
            return cls(sql=sql.group(2).strip())
        elif action_format == 'json':
            action_dict: dict = json.loads(action_text.strip())
            if action_dict.get('action_type', '') != 'GenerateSQL':
                raise MismatchedActionError("Failed to parse GenerateSQL action from the response.")
        
            if action_dict.get('sql', None) is not None:
                sql = str(action_dict['sql']).strip()
            else:
                parameters = action_dict.get('parameters', {})
                if type(parameters) != dict:
                    raise ParseActionError("Failed to parse the SQL query from the response.")
                sql = parameters.get('sql', None)

            if not sql:
                raise ParseActionError("Failed to parse the SQL query from the response.")
            return cls(sql=sql)
        elif action_format == 'xml':
            action_dict = xmltodict.parse(action_text.strip())['action']
            if action_dict.get('action_type', '') != 'GenerateSQL':
                raise MismatchedActionError("Failed to parse GenerateSQL action from the response.")
        
            if action_dict.get('sql', None) is not None:
                sql = str(action_dict['sql']).strip()
            else:
                parameters = action_dict.get('parameters', {})
                if type(parameters) != dict:
                    raise ParseActionError("Failed to parse the SQL query from the response.")
                sql = parameters.get('sql', None)

            if not sql:
                raise ParseActionError("Failed to parse the SQL query from the response.")
            return cls(sql=sql)
        else:
            raise ValueError(f"Action format {action_format} not supported for GenerateSQL.")


@dataclass
class GenerateAnswer(Action):

    answer: str # final answer, required

    def execute(self, env: gym.Env, **kwargs) -> str:
        """ Return the final answer as the observation string.
        """
        return self.answer


    @classmethod
    def _specification(cls, action_format = 'markdown') -> str:
        if action_format == 'markdown':
            action_format = """GenerateAnswer:\n```txt\nfinal answer\n```"""
            use_case = """GenerateAnswer:\n```txt\n42\n```"""
        elif action_format == 'json':
            action_format = '{\n    "action_type": "GenerateAnswer",\n    "parameters": {\n        "answer": Any // final answer, required\n    }\n}'
            example = {
                "action_type": "GenerateAnswer",
                "parameters": {
                    "answer": 42
                }
            }
            use_case = json.dumps(example, indent=4)
        elif action_format == 'xml':
            action_format = """<action><action_type>GenerateAnswer</action_type><parameters><answer>final answer, this is a required tag and the answer format depends on concrete input question</answer></parameters></action>"""
            use_case = """<action><action_type>GenerateAnswer</action_type><parameters><answer>42</answer></parameters></action>"""
        else:
            raise ValueError(f"Action format {action_format} not supported for GenerateAnswer.")
        specification = f"""
### Action Type
GenerateAnswer

### Action Format
{action_format}

### Description
When you take this action, the retrieved results suffice to answer the user question. Please strictly adhere to the answer format of the current question.

### Observation
There is no observation for this terminal action, since it indicates the completion of the task and end of the interaction.

### Use Case
{use_case}
"""
        return specification


    def serialize(self, action_format = 'markdown') -> str:
        """ Serialize the action into a string according to the specified action_format.
        Attention: please conform to the specification.
        """
        action_str = ''
        if hasattr(self, 'thought') and self.thought is not None:
            action_str += f'[Thought]: {self.thought}\n'
        
        action_str += '[Action]: '
        if action_format == 'markdown':
            action_str += f"GenerateAnswer:\n```txt\n{self.answer}\n```"
        elif action_format == 'json':
            action_str += json.dumps({
                "action_type": "GenerateAnswer",
                "parameters": {
                    "answer": self.answer
                }
            }, indent=4)
        elif action_format == 'xml':
            action_str += f"<action><action_type>GenerateAnswer</action_type><parameters><answer>{self.answer}</answer></parameters></action>"
        else:
            raise ValueError(f"Action format {action_format} not supported for GenerateAnswer.")
        return action_str

    @classmethod
    def _parse(cls, action_text: str, action_format: str = 'markdown') -> Action:
        """ Try to parse the Action object from the raw text, if failed, raise an Exception.
        Attention: please conform to the specification.
        """
        if action_format == 'markdown':
            action_type = re.search(r"(.*?):\s*```", action_text.strip())
            if action_type is None or action_type.group(1).strip() != 'GenerateAnswer:':
                raise MismatchedActionError("Failed to parse GenerateAnswer action from the response.")
            
            answer = re.search(r"GenerateAnswer:\s*```(txt)?\s*(.*?)\s*```", action_text.strip(), flags=re.DOTALL)
            if answer is None:
                raise ParseActionError("Failed to parse the final answer from the response.")
            return cls(answer=answer.group(2).strip())
        elif action_format == 'json':
            action_dict: dict = json.loads(action_text.strip())
            if action_dict.get('action_type', '') != 'GenerateAnswer':
                raise MismatchedActionError("Failed to parse GenerateAnswer action from the response.")

            if action_dict.get('answer', None) is not None:
                answer = str(action_dict['answer']).strip()
            else:
                parameters = action_dict.get('parameters', {})
                if type(parameters) != dict:
                    raise ParseActionError("Failed to parse the final answer from the response.")
                answer = parameters.get('answer', None)

            if not answer:
                raise ParseActionError("Failed to parse the final answer from the response.")
            return cls(answer=answer)
        elif action_format == 'xml':
            action_dict = xmltodict.parse(action_text.strip())['action']
            if action_dict.get('action_type', '') != 'GenerateAnswer':
                raise MismatchedActionError("Failed to parse GenerateAnswer action from the response.")

            if action_dict.get('answer', None) is not None:
                answer = str(action_dict['answer']).strip()
            else:
                parameters = action_dict.get('parameters', {})
                if type(parameters) != dict:
                    raise ParseActionError("Failed to parse the final answer from the response.")
                answer = parameters.get('answer', None)

            if not answer:
                raise ParseActionError("Failed to parse the final answer from the response.")
            return cls(answer=answer)
        else:
            raise ValueError(f"Action format {action_format} not supported for GenerateAnswer.")
    
    @property
    def done(self) -> bool:
        return True
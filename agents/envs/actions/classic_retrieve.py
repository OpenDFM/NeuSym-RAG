#coding=utf8
import copy
from agents.envs.actions import Action, Observation, RetrieveFromVectorstore
from dataclasses import dataclass, field
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any


_DEFAULT_VALUES: Dict[str, Any] = {
    "collection_name": "text_sentence_transformers_all_minilm_l6_v2",
    "table_name": "chunks",
    "column_name": "text_content",
    "filter": ""
}


@dataclass
class ClassicRetrieve(Action):
    query: str = field(default='', repr=True) # query string for retrieving the context, required
    limit: int = field(default=5, repr=True) # limit the number of retrieved contexts, optional

    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the vectorstore search result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 10, # maximum rows to display in the output
        "max_tokens": 5000, # maximum tokens to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 600 # the maximum timeout for the vectorstore search is 10 minutes
    }, repr=False)

    _default_values: Dict[str, str] = field(default_factory=lambda: _DEFAULT_VALUES, repr=False)

    @classmethod
    def set_default(cls, **kwargs) -> None:
        global _DEFAULT_VALUES
        for key, value in kwargs.items():
            if key in _DEFAULT_VALUES:
                _DEFAULT_VALUES[key] = value
        return


    def __post_init__(self) -> None:
        self._default_values = copy.deepcopy(_DEFAULT_VALUES)
        return


    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the context from the environment.
        """
        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update output kwargs if exists

        return RetrieveFromVectorstore(
            query=self.query, 
            collection_name=self._default_values['collection_name'],
            table_name=self._default_values['table_name'],
            column_name=self._default_values['column_name'],
            filter=self._default_values['filter'],
            limit=self.limit
        ).execute(env, **output_kwargs)
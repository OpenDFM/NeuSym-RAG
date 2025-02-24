#coding=utf8
from agents.envs.actions import Action, Observation, RetrieveFromVectorstore
from dataclasses import dataclass, field
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any


@dataclass
class ClassicRetrieve(Action):
    query: str = field(default='', repr=True) # query string for retrieving the context, required
    limit: int = field(default=5, repr=True) # limit the number of retrieved contexts, optional
    collection_name: str = field(default='text_sentence_transformers_all_minilm_l6_v2', repr=False)
    table_name: str = field(default='chunks', repr=False)
    column_name: str = field(default='text_content', repr=False)
    filter: str = field(default='', repr=False)

    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the vectorstore search result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 10, # maximum rows to display in the output
        "max_tokens": 5000, # maximum tokens to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 600 # the maximum timeout for the vectorstore search is 10 minutes
    }, repr=False)

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the context from the environment.
        """
        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists

        return RetrieveFromVectorstore(
            query=self.query, 
            collection_name=self.collection_name,
            table_name=self.table_name,
            column_name=self.column_name,
            filter=self.filter,
            limit=self.limit
        ).execute(env, **output_kwargs)
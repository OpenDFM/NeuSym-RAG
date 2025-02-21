#coding=utf8
from agents.envs.actions import Action, Observation, RetrieveFromVectorstore
from dataclasses import dataclass, field
import json, os
import pandas as pd
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any


@dataclass
class ClassicRetrieve(Action):
    query: str = field(default='', repr=True) # query string for retrieving the context, required

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the context from the environment.
        """
        return RetrieveFromVectorstore(
            query=self.query, 
            collection_name="text_sentence_transformers_all_minilm_l6_v2",
            table_name="chunks",
            column_name="text_content",
            filter="",
            limit=4
        ).execute(env, **kwargs)
## Agent Framework for PDF Understanding and QA


### Framework

The entire agent framework can be splitted into 4 parts under folder `agents/`:

```txt
- agents/
    - envs/ # responsible for gym-like environments and action specification/parse/serialization/execution
        - actions.py # all allowable Action classes, must inherit class Action, and implement four methods: _specification, _parse, serialize, and execute.
        - env_base.py # wrapper for OpenAI gym, automatically parse input text based on allowable actions
        - text2sql_env.py # class Text2SQLEnv, allowable actions [GenerateSQL, GenerateAnswer]
        - text2vec_env.py # class Text2VecEnv
    - models/ # responsible for calling LLMs
        - llm_cache.py # record LLM cache to save budget and time
        - llm_base.py # basic class for different llm clients, each sub-class must implement pre-defined interfaces
        - llm_gpt.py # maintain OpenAI API calls
    - prompts/ # different prompt templates
        - agent_prompt.py # different agent/interaction method, e.g., ReAct, Plan-and-Solve
        - system_prompt.py # for different interactive environments and task input
        - database_schema_prompt.py # for database and vectorstore serialization
    - frameworks/ # agent frameworks which combine all stuff above, e.g., environments, models, and prompts
        - text2sql_rag.py # agentic baseline: text-to-SQL symbolic retrieval
        - text2vec_rag.py # agentic baseline: text-to-vector neural retrieval
```


### Running scripts

The predicted result is saved in folder `results/` in `.jsonl` format, and the log history is saved in folder `logs/` in `.log` format.

1. Text-to-SQL with interactive environment baseline + different agent frameworks:
    - `react`: [ReAct](https://arxiv.org/pdf/2210.03629)

```sh
python scripts/text2sql_baseline.py --dataset pdfvqa --database biology_paper --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt4-o-mini --max_turn 10
```
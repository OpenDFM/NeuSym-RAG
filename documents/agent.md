## Agent Framework for PDF Understanding and QA


### Framework

The entire agent framework can be splitted into 5 parts under folder `agents/`:

```txt
- agents/
    - envs/ # responsible for gym-like environments, e.g., Text2SQLEnv will execute the SQL and provide serialized output table
        - env_base.py # wrapper for OpenAI gym, must implement action-to-string function "serialize_action"
        - text2sql_env.py # class Text2SQLEnv, two allowable actions `GenerateSQL` and `GenerateAnswer`
    - models/ # responsible for calling LLMs
        - llm_cache.py # record LLM cache to save budget and time
        - llm_base.py # basic class for different llm clients, each sub-class must implement pre-defined interfaces
        - llm_gpt.py # maintain OpenAI API calls
    - parsers/ # responsible for parsing LLM response text into action dict
        - base_output_parser.py # only one interface `parse()` which needs to be implemented
        - text2sql_output_parser.py # for text-to-SQL baselines, how to parse the actions from raw string
    - prompts/ # record different prompt templates, including system message, database schema, action and observation space, and interaction procedure/agent methods
    - frameworks/ # agent frameworks which combine all stuff above, e.g., prompts, parsers, llm clients and environments
        - text2sql_rag.py # baseline: text-to-SQL symbolic retrieval
        - naive_vector_rag.py # baseline: naive vector-based neural retrieval (no interaction, classic pre-retrieval)
        - vector_rag.py # baseline: treat vector-based retrieval as an agentic task
```


### Running scripts

Text-to-SQL baseline + [ReAct](https://arxiv.org/pdf/2210.03629) framework:

```sh
python scripts/text2sql_react_baseline.py --dataset pdfvqa --database biology_paper --test_data test_data_sample.jsonl
```

- predicted results will be saved under folder `results/` with filename `{dataset}_text2sql_react_{llm}.json`
- log history will be saved under folder `logs/` with filename `text2sql_react_baseline-{YYYY-MM-DD}.log`
## Agent Framework for PDF Understanding and QA


### Framework

The entire agent framework can be splitted into 4 parts under folder `agents/`:

```txt
- agents/
    - envs/ # responsible for gym-like environments and action specification/parse/serialization/execution
        - actions/
            - action.py # base Action class, which implements `get_action_space_prompt`, `parse_action` and `convert_to_message` functions
            - actions.json # define each action type, description, parameters and use cases in JSON format
            - observation.py # Observation class, which wraps the action execution result
            - ... # one .py file for each distinct action sub-class, must define all parameter fields and implement the `execute` function
        - env_base.py # wrapper for OpenAI gym, automatically parse input text based on allowable actions
        - text2sql_env.py # maintain connection to database
        - text2vec_env.py # maintain connection to vectorstore
        - hybrid_env.py # maintain connections to both database and vectorstore
    - models/ # responsible for calling LLMs
        - llm_cache.py # record LLM cache to save budget and time
        - llm_base.py # basic class for different llm clients, each sub-class must implement pre-defined interfaces
        - llm_gpt.py # maintain OpenAI API calls
    - prompts/ # different prompt templates
        - agent_prompt.py # different agent/interaction method, e.g., ReAct
        - system_prompt.py # for different interactive environments and task input
        - schema_prompt.py # for database and vectorstore serialization
        - task_prompt.py # for concrete datasets, specify the output formatting requirements
    - frameworks/ # agent frameworks which combine all stuff above, e.g., environments, models, and prompts
        - agent_base.py # base class
        - text2sql_rag.py # agentic baseline: text-to-SQL symbolic retrieval
        - text2vec_rag.py # agentic baseline: text-to-vector neural retrieval
        - two_stage_text2sql_rag.py # baseline: first generate SQL, then generate answer
        - two_stage_text2vec_rag.py # baseline: first generate RetrieveFromVectorstore action, then generate answer
        - classic_rag.py # baseline: pre-fetch the relevant context, then generate the answer based on retrieved docs (calling LLM once)
```


### Running scripts

Here are some common arguments:
- dataset and database:
    - `pdfvqa` -> `biology_paper`
    - `tatdqa` -> `financial_report`
- the predicted result and log history are both saved in folder `results/`
- `action_format`: chosen from [`json`, `markdown`, `xml`, `yaml`], by default is `json`
- `output_format`: only used in action `RetrieveFromDatabase` and `RetrieveFromVectorstore`, chosen from [`markdown`, `json`, `html`, `string`], by default is `markdown` table. Currently, you need to modify this parameter in the action file, e.g., `agents/envs/actions/retrieve_from_database.py`

1. Text-to-SQL with interactive database environment baseline:
```sh
python scripts/text2sql_baseline.py --dataset pdfvqa --database biology_paper --test_data test_data_sample.jsonl --action_format json --agent_method 'react' --llm gpt-4o-mini --max_turn 10
python scripts/text2sql_baseline.py --dataset tatdqa --database financial_report --test_data test_data_sample.jsonl --action_format json --agent_method 'react' --llm gpt-4o-mini --max_turn 10
```

2. Text-to-vector with interactive vectorstore environment baseline:
```sh
python scripts/text2vec_baseline.py --dataset pdfvqa --vectorstore biology_paper --test_data test_data_sample.jsonl --action_format json --agent_method 'react' --llm gpt-4o-mini --max_turn 10
python scripts/text2vec_baseline.py --dataset tatdqa --vectorstore financial_report --test_data test_data_sample.jsonl --action_format json --agent_method 'react' --llm gpt-4o-mini --max_turn 10
```

3. Two stage text-to-SQL (non-interactive) baseline:

```sh
python scripts/two_stage_text2sql_baseline.py --dataset pdfvqa --database biology_paper --test_data test_data_sample.jsonl --agent_method 'two_stage_text2sql' --llm gpt-4o-mini --max_turn 2
python scripts/two_stage_text2sql_baseline.py --dataset tatdqa --database financial_report --test_data test_data_sample.jsonl --agent_method 'two_stage_text2sql' --llm gpt-4o-mini --max_turn 2
```

4. Two stage text-to-vector (non-interactive) baseline:

```sh
python scripts/two_stage_text2vec_baseline.py --dataset pdfvqa --vectorstore biology_paper --test_data test_data_sample.jsonl --agent_method 'two_stage_text2vec' --llm gpt-4o-mini --max_turn 2
python scripts/two_stage_text2vec_baseline.py --dataset tatdqa --vectorstore financial_report --test_data test_data_sample.jsonl --agent_method 'two_stage_text2vec' --llm gpt-4o-mini --max_turn 2
```

5. Classic RAG (pre-fetch the context and call LLM once) baseline:
- `--collection_name` can be changed to any other embedding models (see `data/vectorstore/*/*.json` for all available collections)
- if `--table_name` and `--column_name` are not specified, all encodable text content will be used by default (see `data/database/*/*.json` for all available encodable columns)
- `--limit` restricts the number of returned chunks
```sh
python scripts/classic_rag_baseline.py --dataset pdfvqa --vectorstore biology_paper --test_data test_data_sample.jsonl --agent_method classic_rag --llm gpt-4o-mini --max_turn 1 --collection_name text_bm25_en --table_name chunks --column_name text_content --limit 2
python scripts/classic_rag_baseline.py --dataset tatdqa --vectorstore financial_report --test_data test_data_sample.jsonl --agent_method classic_rag --llm gpt-4o-mini --max_turn 1 --collection_name text_bm25_en --table_name chunks --column_name text_content --limit 2
```
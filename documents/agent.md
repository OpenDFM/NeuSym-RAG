# Agent Baselines


## Folder Structure

<details><summary>👇🏻 Click to preview the <code>agents</code> module</summary>

```txt
agents/
├── envs/
│   ├── __init__.py
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── action.py
│   │   ├── actions.json
│   │   ├── calculate_expr.py
│   │   ├── classic_retrieve.py
│   │   ├── error_action.py
│   │   ├── generate_answer.py
│   │   ├── observation.py
│   │   ├── retrieve_from_database.py
│   │   ├── retrieve_from_vectorstore.py
│   │   └── view_image.py
│   ├── classic_env.py
│   ├── env_base.py
│   ├── graph_env.py
│   ├── hybrid_env.py
│   ├── neural_env.py
│   └── symbolic_env.py
├── frameworks/
│   ├── __init__.py
│   ├── agent_base.py
│   ├── classic_rag_agent.py
│   ├── iterative_classic_rag_agent.py
│   ├── iterative_neu_rag_agent.py
│   ├── iterative_sym_rag_agent.py
│   ├── neusym_rag_agent.py
│   ├── trivial_baseline.py
│   ├── two_stage_hybrid_rag_agent.py
│   ├── two_stage_neu_rag_agent.py
│   └── two_stage_sym_rag_agent.py
├── models/
│   ├── __init__.py
│   ├── llm_base.py
│   ├── llm_cache.py
│   ├── llm_gpt.py
│   └── llm_vllm.py
└── prompts/
    ├── __init__.py
    ├── agent_prompt.py
    ├── hint_prompt.py
    ├── schema_prompt.py
    ├── system_prompt.py
    └── task_prompt.py
```

</details>

The entire `agents` package can be splitted into 4 sub-modules:
- 🌏 `envs`: responsible for `gym`-like environments and action/observation space management (e.g., action specification, action parsing, serialization to messages, and execution upon the backend)
- 📭 `models`: responsible for calling LLMs. We implement the unified interface for both closed-source ([`GPTClient`](../agents/models/llm_gpt.py)) and open-source LLMs ([`VLLMClient`](../agents/models/llm_vllm.py)), along with a SQLite cache ([`Sqlite3CacheProvider`](../agents/models/llm_cache.py)) to store historical responses
- 📜 `prompts`: responsible for different prompt templates, e.g., system prompt, agent prompt, task prompt, schema prompt and hint prompt.
- ⛩️ `frameworks`: responsible for different agentic frameworks. Each baseline method inherits from the base `AgentBase` class and implements the `interact` function.


## Overview of Agent Baselines

<p align="center">
  <img src="../assets/agent_baselines.png" alt="Agent Baselines" width="95%">
  <br>
  <em>The Comparison of Different Agent Baselines</em>
</p>

Here are the checklist of all different agent baselines:

| method                         | neural | symbolic | multi-view | multi-turn | scripts (`scripts`) | environments (`agents.envs`) | frameworks (`agents.frameworks`) |
|:------------------------------ |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Trivial: question only         | ❌ | ❌ | ❌ | ❌ | [`trivial_question_only_baseline`](../scripts/trivial_question_only_baseline.py) | [`AgentEnv`](../agents/envs/env_base.py) | [`TrivialBaselineAgent`](../agents/frameworks/trivial_baseline.py) |
| Trivial: title + abstract      | ❌ | ❌ | ❌ | ❌ | [`trivial_title_with_abstract_baseline`](../scripts/trivial_title_with_abstract_baseline.py) | [`AgentEnv`](../agents/envs/env_base.py) | [`TrivialBaselineAgent`](../agents/frameworks/trivial_baseline.py) |
| Trivial: full-text with cutoff | ❌ | ❌ | ❌ | ❌ | [`trivial_full_text_with_cutoff_baseline`](../scripts/trivial_full_text_with_cutoff_baseline.py) | [`AgentEnv`](../agents/envs/env_base.py) | [`TrivialBaselineAgent`](../agents/frameworks/trivial_baseline.py) |
| Classic-RAG                    | ✅ | ❌ | ❌ | ❌ | [`classic_rag_baseline`](../scripts/classic_rag_baseline.py) | [`ClassicRAGEnv`](../agents/envs/classic_env.py) | [`ClassicRAGAgent`](../agents/frameworks/classic_rag_agent.py) |

| Iterative Classic-RAG          | ✅ | ❌ | ❌ | ✅ |
| Two-stage Neu-RAG              | ✅ | ❌ | ✅ | ❌ |
| Iterative Neu-RAG              | ✅ | ❌ | ✅ | ✅ |
| Two-stage Sym-RAG              | ❌ | ✅ | ✅ | ❌ |
| Iterative Sym-RAG              | ❌ | ✅ | ✅ | ✅ |
| Two-stage Graph-RAG            | ✅ | ❌ | ✅ | ❌ |
| Iterative Graph-RAG            | ✅ | ❌ | ✅ | ✅ |
| Two-stage Hybrid-RAG           | ✅ | ✅ | ✅ | ❌ |
| **NeuSym-RAG**                 | ✅ | ✅ | ✅ | ✅ |

> **❗️ Note:** Code for Two-stage Graph-RAG and Iterative Graph-RAG are preparing.


Here are some common arguments:
- dataset and database:
    - `pdfvqa` -> `biology_paper`
    - `tatdqa` -> `financial_report`
    - `airqa` -> `ai_research`
- the predicted result and log history are both saved in folder `results/`
- `action_format`: chosen from [`json`, `markdown`, `xml`, `yaml`], by default is `markdown`
- `output_format`: only used in actions `RetrieveFrom*`, chosen from [`markdown`, `json`, `html`, `string`], by default is `json`. Currently, you need to modify this parameter in the action file, e.g., `agents/envs/actions/retrieve_from_database.py`

0. Hybrid neural symbolic interactive retrieval framework:
```sh
python scripts/hybrid_neural_symbolic_rag.py --dataset pdfvqa --database biology_paper --vectorstore biology_paper --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt-4o-mini --max_turn 15
python scripts/hybrid_neural_symbolic_rag.py --dataset tatdqa --database financial_report --vectorstore financial_report --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt-4o-mini --max_turn 15
```

1. Text-to-SQL with interactive database environment baseline:
```sh
python scripts/text2sql_baseline.py --dataset pdfvqa --database biology_paper --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt-4o-mini --max_turn 10
python scripts/text2sql_baseline.py --dataset tatdqa --database financial_report --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt-4o-mini --max_turn 10
```

2. Text-to-vector with interactive vectorstore environment baseline:
```sh
python scripts/text2vec_baseline.py --dataset pdfvqa --vectorstore biology_paper --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt-4o-mini --max_turn 10
python scripts/text2vec_baseline.py --dataset tatdqa --vectorstore financial_report --test_data test_data_sample.jsonl --action_format markdown --agent_method 'react' --llm gpt-4o-mini --max_turn 10
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
- the arguments `--table_name` and `--column_name` must be specified (see `data/database/*/*.json` for all available encodable columns)
- `--limit` restricts the number of returned chunks
```sh
python scripts/classic_rag_baseline.py --dataset pdfvqa --vectorstore biology_paper --test_data test_data_sample.jsonl --agent_method classic_rag --llm gpt-4o-mini --max_turn 1 --collection_name text_sentence_transformers_all_minilm_l6_v2 --table_name chunks --column_name text_content --limit 2
python scripts/classic_rag_baseline.py --dataset tatdqa --vectorstore financial_report --test_data test_data_sample.jsonl --agent_method classic_rag --llm gpt-4o-mini --max_turn 1 --collection_name text_sentence_transformers_all_minilm_l6_v2 --table_name chunks --column_name text_content --limit 2
```
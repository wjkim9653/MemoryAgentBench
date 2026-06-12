# Qwen3-4B Table 3 Reproduction Progress

This note records the partial reproduction of MemoryAgentBench Table 3 using
`Qwen/Qwen3-4B-Instruct-2507` as the generation model. It is intended as the
reference for adding more rows and filling missing columns later.

## Scope

Target paper table: Table 3, "Overall Performance Comparison".

Rows currently reproduced:

- `Qwen3-4B LongCtx`: long-context agent with Qwen3-4B as the model.
- `Qwen3-4B BM25`: simple BM25 RAG with Qwen3-4B as the generator.

Rows prepared but not yet run:

- `Qwen3-4B HippoRAG-v2`: structure-augmented RAG with Qwen3-4B as the
  generator and `nvidia/NV-Embed-v2` as the local embedding model.

Rows not yet prepared:

- Embedding RAG agents.
- Agentic memory agents.

Columns intentionally skipped for now:

- `TTL / Recom.`: Recsys requires extra processed data, especially
  `processed_data/Recsys_Redial/entity2id.json`.
- Official `AR / LME(S*)`: requires LLM-as-judge evaluation.
- Official `LRU / Summ.`: requires LLM-as-judge summarization evaluation.

For skipped judge columns, raw generation results were still produced where the
runner includes the dataset. These raw metrics are diagnostic only and should
not be mixed into the official Table 3 score.

## Model And Serving Setup

Model:

```text
Qwen/Qwen3-4B-Instruct-2507
```

Serving style:

```text
OpenAI-compatible vLLM endpoint
```

Representative serve command:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 32768 \
  --dtype auto \
  --gpu-memory-utilization 0.90
```

In later runs the endpoint reported support for a larger context window, and the
local agent config used:

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
tokenizer_model: Qwen/Qwen3-4B-Instruct-2507
provider: openai_compatible
api_base: http://127.0.0.1:8000/v1
api_key: EMPTY
input_length_limit: 262144
buffer_length: 4000
```

Important interpretation detail: long-context runs are not guaranteed to expose
the full benchmark context to the model. The implementation trims context to:

```text
agent_config.input_length_limit - agent_config.buffer_length - dataset_config.generation_max_length
```

If the benchmark context is longer than that budget, the older prefix is dropped
and the model sees the tail of the memory context. This matches the repo's
long-context baseline behavior and should be reported as a tail-truncated
long-context baseline.

## Configs And Runners

Table 3 reproduction manifests, excluding Recsys:

```text
bash_files/configs/qwen3_4b_vllm_longcontext_table3.txt
bash_files/configs/qwen3_4b_vllm_bm25_table3.txt
bash_files/configs/qwen3_4b_vllm_hipporag_table3.txt
```

Execution scripts:

```text
bash_files/sh/run_qwen3_4b_vllm_longcontext_table3.sh
bash_files/sh/run_qwen3_4b_vllm_bm25_table3.sh
bash_files/sh/run_qwen3_4b_vllm_hipporag_table3.sh
```

HippoRAG smoke runner:

```text
bash_files/configs/qwen3_4b_vllm_hipporag_smoke.txt
bash_files/sh/run_qwen3_4b_vllm_hipporag_smoke.sh
```

Commands used on the GPU server:

```bash
bash bash_files/sh/run_qwen3_4b_vllm_longcontext_table3.sh 2>&1 | tee outputs/qwen3-4b-vllm-longcontext-table3.log
bash bash_files/sh/run_qwen3_4b_vllm_bm25_table3.sh 2>&1 | tee outputs/qwen3-4b-vllm-bm25-table3.log
```

Recommended HippoRAG smoke command on the GPU server:

```bash
bash bash_files/sh/run_qwen3_4b_vllm_hipporag_smoke.sh 2>&1 | tee outputs/qwen3-4b-vllm-hipporag-smoke.log
```

If the run fails with:

```text
ModuleNotFoundError: No module named 'igraph'
```

install the missing graph dependency in the active uv environment:

```bash
uv pip install "igraph>=0.11,<0.12"
python - <<'PY'
import igraph
print("igraph", igraph.__version__)
from methods.hipporag import HippoRAG
print("HippoRAG import OK")
PY
```

If the run fails with:

```text
ModuleNotFoundError: No module named 'gritlm'
```

do not install `gritlm` for the NV-Embed-v2 run. `gritlm` is only needed for the
unused GritLM embedding backend. The local code lazy-loads optional embedding
backends so that `nvidia/NV-Embed-v2` does not require `gritlm`.

The same principle applies to the offline vLLM OpenIE path: the current online
OpenIE config calls the existing OpenAI-compatible server over HTTP, so the local
benchmark environment does not need to import the offline `vllm` backend.

If the run fails with:

```text
ModuleNotFoundError: No module named 'hipporag'
```

this comes from the vendored HippoRAG prompt loader using upstream package names.
The local code should import templates via `methods.hipporag.prompts.templates`,
not an installed `hipporag` package. Pull the patch that updates
`methods/hipporag/prompts/prompt_template_manager.py`; do not install the
external `hipporag` package into this benchmark environment.

For the current Qwen3-4B + HippoRAG-v2 + NV-Embed-v2 setup, the non-baseline
packages to check are:

```bash
uv pip install "igraph>=0.11,<0.12" einops
```

Most other HippoRAG imports are already part of the base benchmark environment:
`torch`, `transformers`, `accelerate`, `numpy`, `pandas`, `tqdm`, `openai`,
`httpx`, `filelock`, `packaging`, `pydantic`, and `tenacity`.

Recommended HippoRAG Table 3 command after the smoke run succeeds:

```bash
bash bash_files/sh/run_qwen3_4b_vllm_hipporag_table3.sh 2>&1 | tee outputs/qwen3-4b-vllm-hipporag-table3.log
```

Both HippoRAG scripts default to:

```bash
CUDA_VISIBLE_DEVICES=1
```

This is intentional. The benchmark process loads `nvidia/NV-Embed-v2` locally for
HippoRAG indexing, while Qwen generation and OpenIE calls are sent over HTTP to
the existing vLLM endpoint at `http://127.0.0.1:8000/v1`.

Aggregator:

```bash
python3 utils/aggregate_qwen_table3.py
```

## Table 3 Column Mapping

The following mapping is used for this reproduction.

| Table 3 column | Dataset config | Result sub_dataset | Metric used now |
|---|---|---|---|
| `AR / SH-QA` | `Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml` | `ruler_qa1_197K` | `substring_exact_match` |
| `AR / MH-QA` | `Accurate_Retrieval/Ruler/QA/Ruler_qa2_421k.yaml` | `ruler_qa2_421K` | `substring_exact_match` |
| `AR / LME(S*)` | `Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml` | `longmemeval_s*` | pending official LLM judge |
| `AR / EventQA` | `Accurate_Retrieval/EventQA/Eventqa_full.yaml` | `eventqa_full` | `substring_exact_match` |
| `TTL / MCC` | five `Test_Time_Learning/ICL/*.yaml` configs | `icl_*` | mean `exact_match` |
| `TTL / Recom.` | `Test_Time_Learning/Recsys/Recsys_redial_full.yaml` | `recsys_redial_full` | skipped, data missing |
| `LRU / Summ.` | `Long_Range_Understanding/InfBench_sum.yaml` | `infbench_sum_eng_shots2` | pending official LLM judge |
| `LRU / DetQA` | `Long_Range_Understanding/Detective_QA.yaml` | `detective_qa` | `exact_match` |
| `SF / FC-SH` | `Conflict_Resolution/Factconsolidation_sh_262k.yaml` | `factconsolidation_sh_262k` | `substring_exact_match` |
| `SF / FC-MH` | `Conflict_Resolution/Factconsolidation_mh_262k.yaml` | `factconsolidation_mh_262k` | `substring_exact_match` |

The official Table 3 averages cannot be fully computed until Recsys and judge
metrics are available. The current `AR partial` excludes `LME(S*)`; the current
`LRU official` and `Overall official` are intentionally left pending.

## Current Partial Results

Generated by:

```bash
python3 utils/aggregate_qwen_table3.py
```

| Row | SH-QA | MH-QA | LME(S*) raw | EventQA | AR partial | MCC strict | TTL official | InfBench raw F1 | InfBench raw RLsum | DetQA | LRU official | FC-SH | FC-MH | SF Avg. | Overall official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B LongCtx | 33.0 | 26.0 | 15.0 | 47.4 | 35.5 | 21.4 | PENDING | 31.2 | 30.1 | 0.0 | PENDING | 29.0 | 0.0 | 14.5 | PENDING |
| Qwen3-4B BM25 | 60.0 | 42.0 | 26.7 | 53.0 | 51.7 | 0.4 | PENDING | 32.8 | 31.8 | 0.0 | PENDING | 40.0 | 3.0 | 21.5 | PENDING |
| Qwen3-4B HippoRAG-v2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Relaxed Diagnostics

These diagnostics are not part of the official Table 3 reproduction. They are
included because Qwen often returns labels as `label: 18` or DetectiveQA answers
inside JSON objects, while official `exact_match` is strict.

| Row | Banking numeric | Clinic numeric | NLU numeric | TREC-C numeric | TREC-F numeric | MCC numeric | DetQA answer-field |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B LongCtx | 94.0 | 87.0 | 80.0 | 74.0 | 67.0 | 80.4 | 73.2 |
| Qwen3-4B BM25 | 91.0 | 87.0 | 79.0 | 77.0 | 54.0 | 77.6 | 67.6 |

## Observations

- BM25 improves retrieval-heavy AR columns over direct long-context prompting:
  `SH-QA` improves from `33.0` to `60.0`, `MH-QA` from `26.0` to `42.0`, and
  `EventQA` from `47.4` to `53.0`.
- SF remains the weakest axis. `FC-MH` is `0.0` for LongCtx and `3.0` for BM25,
  suggesting that retrieval alone does not solve multi-hop conflict resolution.
- Strict TTL and DetQA scores severely understate answer correctness for Qwen
  because of output-format mismatch. Keep official strict numbers separate from
  relaxed diagnostics.
- `InfBench` and `LongMemEval` raw metrics are useful for debugging only. They
  should not be used as official Table 3 values without the judge scripts.
- HippoRAG-v2 is expected to be much slower than BM25. It performs OpenIE over
  memory chunks, embeds chunks/entities/facts with `nvidia/NV-Embed-v2`, builds a
  graph, and then performs graph-aware retrieval before QA. The likely bottleneck
  is indexing, especially on 262k contexts.

## Advanced Agent Readiness

### HippoRAG-v2

HippoRAG-v2 is the next structure-augmented RAG row to run. The local
implementation already had OpenAI-compatible hooks in `methods/hipporag`, but
the top-level benchmark wrapper did not pass the Qwen/vLLM endpoint into
HippoRAG. The Qwen config now wires:

```yaml
provider: openai_compatible
api_base: http://127.0.0.1:8000/v1
api_key: EMPTY
hipporag_embedding_model: nvidia/NV-Embed-v2
hipporag_max_new_tokens: 2048
hipporag_qa_top_k: 10
```

The implementation path is:

```text
agent.py::_handle_hippo_rag
  -> methods.hipporag.HippoRAG
  -> methods.hipporag.information_extraction.OpenIE
  -> methods.hipporag.llm.CacheOpenAI, using the vLLM endpoint
  -> methods.hipporag.embedding_model.NVEmbedV2EmbeddingModel, local GPU
```

`hipporag_max_new_tokens` is intentionally separate from
`dataset_config.generation_max_length`. HippoRAG uses the same LLM for OpenIE
index construction and QA, so using short benchmark answer lengths such as 10
tokens would truncate NER/triple extraction.

### Agentic Memory Candidates

`MemGPT` in the paper maps most closely to the repo's Letta implementation:

```text
configs/agent_conf/RAG_Agents/gpt-4o-mini/Agentic_memory_gpt-4o-mini-letta.yaml
agent.py::_initialize_letta_agent
agent.py::_handle_letta_agent
```

This is the most faithful Agentic Memory row candidate, but it is not the
simplest next implementation. It depends on Letta client/server state, local
agent databases, and its own LLM/embedding configuration surface. It should be
prepared after HippoRAG is validated.

`MIRIX` does not currently appear to have a first-class implementation in this
repo, so reproducing that row would require adding an external integration rather
than adapting an existing agent.

`Self-RAG` exists in `methods/self_rag.py`, but the current code is tied to
`ChatOpenAI`, `OpenAIEmbeddings`, and LangChain structured-output calls. It is
probably less infrastructure-heavy than Letta, but Qwen/vLLM support requires
more prompt/output-format work than HippoRAG because multiple internal judges
route/rewrite/grade with structured outputs.

## Pending Items

- Find and install Recsys processed data:

```text
processed_data/Recsys_Redial/entity2id.json
```

- Run LLM judge for `LongMemEval` when cost is acceptable:

```bash
python llm_based_eval/longmem_qa_evaluate.py --evaluated_method qwen3-4b-vllm --dataset 'longmemeval_s*'
python llm_based_eval/longmem_qa_evaluate.py --evaluated_method qwen3-4b-vllm-bm25 --dataset 'longmemeval_s*'
```

- Run summarization judge for `InfBench_sum` when cost is acceptable.
- Run the HippoRAG smoke script, then the HippoRAG Table 3 script if indexing
  and QA complete successfully.
- Prepare one Agentic Memory row after HippoRAG. Letta is most faithful to
  `MemGPT`; Self-RAG is the fallback if Letta setup turns out to be too brittle.

# A Systematic Study of Pseudo-Relevance Feedback with LLMs

This is the code for the paper: [A Systematic Study of Pseudo-Relevance Feedback with LLMs](https://arxiv.org/abs/2603.11008).

## Abstract

Pseudo-relevance feedback (PRF) methods based on large language models (LLMs) can be understood through two main design choices: the source of the feedback text and the model used to incorporate that feedback into the query representation. Prior evaluations often combine these choices, making it difficult to understand the contribution of each component. In this work, we study these dimensions through controlled experiments across low-resource BEIR tasks and several LLM-based PRF methods. Our results show that feedback models can substantially affect retrieval effectiveness, LLM-generated feedback offers a strong cost-effective option, and corpus-derived feedback is most useful when candidate documents come from a strong first-stage retriever.

## Repository Layout

```text
.
├── run_hyde.py
├── run_prf.py
├── run_hyde_prf.py
├── run_prf_umbrela.py
├── run_umbrela_hyde.py
├── modules/
└── feedback_methods/
```

Main entry points:

- `run_hyde.py`: LLM-only feedback. Generates HyDE-style hypothetical documents, then applies a feedback model such as `rocchio` or `rm3`.
- `run_prf.py`: Corpus-only pseudo-relevance feedback baselines using the top initially retrieved documents.
- `run_hyde_prf.py`: HyDE-PRF, where the LLM generates feedback conditioned on initially retrieved PRF documents.
- `run_prf_umbrela.py`: UMBRELA-style LLM-judged corpus feedback. The LLM judges retrieved documents, then relevant documents are passed to a feedback model.
- `run_umbrela_hyde.py`: Combined corpus and LLM feedback, using UMBRELA judgments together with HyDE-generated feedback.

Supporting code lives in:

- `modules/`: query loading, retrieval helpers, LLM wrappers, and dataset/index mappings.
- `feedback_methods/`: sparse feedback implementations, including Rocchio, RM3, and HyDE-style feedback.

## Setup

The code depends on Pyserini/Anserini for retrieval and evaluation, vLLM for LLM inference, and Hugging Face Transformers for tokenization/model loading.

Install the core dependencies in your preferred environment:

```bash
pip install pyserini vllm transformers torch tqdm numpy
```

Pyserini requires a working Java installation. If you are using an editable or custom Pyserini checkout, make sure the Python code and Anserini fatjar are compatible. If needed, point Pyserini at the matching Anserini jar directory:

```bash
export ANSERINI_CLASSPATH=/path/to/anserini-fatjar-directory
```

The experiments rely on Pyserini's prebuilt indexes and topics. Make sure Pyserini can download or access the indexes for the datasets you run.

## Supported Datasets

The dataset keys are defined in `modules/index_paths.py`. Supported sparse retrieval datasets include:

```text
covid, news, scifact, fiqa, nfcorpus, dbpedia,
robust04, scidocs, arguana, nq, bioasq, signal1m, climate-fever
```

For TREC-COVID, use:

```bash
--corpus_name covid
```

## Quickstart

Create output/cache directories:

```bash
mkdir -p result_files/covid precomputed_passages precomputed_judgements
```

### HyDE With Rocchio

This runs LLM-only feedback: Qwen generates hypothetical documents, and Rocchio turns those documents into a sparse BM25 query.

```bash
python run_hyde.py \
  --model_path Qwen/Qwen3-14B \
  --num_gpus 1 \
  --corpus_name covid \
  --returned_hits 100 \
  --feedback_mechanism rocchio \
  --encoder bm25 \
  --precomputed_passages precomputed_passages/hyde_covid_model-Qwen3-14B_precomputed_passages \
  --save_precomputed_passages \
  --output_folder result_files/covid
```

To use cached HyDE generations on later runs, omit `--save_precomputed_passages` and keep the same `--precomputed_passages` path.

### Traditional PRF

This uses top-ranked corpus documents as feedback.

```bash
python run_prf.py \
  --corpus_name covid \
  --encoder bm25 \
  --feedback_documents 8 \
  --returned_hits 100 \
  --feedback_mechanism rocchio \
  --output_folder result_files/covid
```

### PRF-UMBRELA

This first retrieves documents, asks the LLM to judge the top hits, and then uses the judged relevant documents as feedback.

```bash
python run_prf_umbrela.py \
  --model_path Qwen/Qwen3-14B \
  --num_gpus 1 \
  --corpus_name covid \
  --hits_judged 10 \
  --returned_hits 100 \
  --feedback_mechanism rocchio \
  --initial_encoder bm25 \
  --final_encoder bm25 \
  --output_folder result_files/covid
```

### HyDE-PRF

This conditions LLM-generated feedback on initially retrieved PRF passages.

```bash
python run_hyde_prf.py \
  --model_path Qwen/Qwen3-14B \
  --num_gpus 1 \
  --corpus_name covid \
  --prf_docs 10 \
  --returned_hits 100 \
  --feedback_mechanism rocchio \
  --initial_encoder bm25 \
  --final_encoder bm25 \
  --output_folder result_files/covid
```

### UMBRELA-HyDE

This combines corpus feedback selected by UMBRELA with HyDE-generated feedback.

```bash
python run_umbrela_hyde.py \
  --model_path Qwen/Qwen3-14B \
  --num_gpus 1 \
  --corpus_name covid \
  --hits_judged 10 \
  --returned_hits 100 \
  --feedback_mechanism rocchio \
  --initial_encoder bm25 \
  --final_encoder bm25 \
  --hyde_precomputed_passages precomputed_passages/hyde_covid_model-Qwen3-14B_precomputed_passages \
  --output_folder result_files/covid
```

## Feedback Mechanisms

For sparse BM25 retrieval, the main feedback mechanisms are:

- `rocchio`: use Rocchio to combine the original query vector with feedback document vectors.
- `rm3`: use RM3-style relevance modeling over feedback documents.

## Outputs and Caches

Runs write TREC-format ranking files to the directory passed with `--output_folder`.

Generated or judged intermediate artifacts are cached as pickle files:

- `precomputed_passages/`: cached HyDE generations.
- `precomputed_passages_hyde_prf/`: cached HyDE-PRF generations.
- `precomputed_judgements/`: cached UMBRELA relevance judgments.

These caches are optional but useful for avoiding repeated LLM inference.

## Citation

If you use this code, please cite:

```bibtex
@article{jedidi2026systematic,
  title={A Systematic Study of Pseudo-Relevance Feedback with LLMs},
  author={Jedidi, Nour and Lin, Jimmy},
  journal={arXiv preprint arXiv:2603.11008},
  year={2026}
}
```


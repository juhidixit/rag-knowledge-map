# QASPER retrieval benchmark

888 answerable questions with annotated evidence paragraphs over 281 full-text arXiv NLP papers (QASPER dev v0.3). Embedder `/home/user/models/all-MiniLM-L6-v2`, reranker `/home/user/models/ms-marco-MiniLM-L-6-v2`. Hardware: x86_64 / x86_64 / 4 vCPU, CPU only. All numbers in %, higher is better.

## Search inside the paper (chat-with-a-PDF)

| Chunking | Retriever | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Evidence recall@5 |
|---|---|---|---|---|---|---|
| fixed-200w | BM25 | 26.9 | 67.3 | 43.3 | 51.1 | 56.2 |
| fixed-200w | Dense (MiniLM) | 30.1 | 72.2 | 47.5 | 55.1 | 60.2 |
| fixed-200w | Hybrid (RRF) | 33.6 | 73.4 | 50.3 | 58.3 | 61.2 |

## Search across all 281 papers

| Chunking | Retriever | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Evidence recall@5 |
|---|---|---|---|---|---|---|

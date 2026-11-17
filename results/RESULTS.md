# QASPER retrieval benchmark

888 answerable questions with annotated evidence paragraphs over 281 full-text arXiv NLP papers (QASPER dev v0.3). Embedder `sentence-transformers/all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`. Hardware: x86_64 / x86_64 / 4 vCPU, CPU only. All numbers in %, higher is better.

## Search inside the paper (chat-with-a-PDF)

| Chunking | Retriever | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Evidence recall@5 |
|---|---|---|---|---|---|---|
| fixed-200w | BM25 | 26.9 | 67.3 | 43.3 | 51.1 | 56.2 |
| fixed-200w | Dense (MiniLM) | 30.1 | 72.2 | 47.5 | 55.1 | 60.2 |
| fixed-200w | Hybrid (RRF) | 33.6 | 73.4 | 50.3 | 58.3 | 61.2 |
| fixed-200w | Hybrid + cross-encoder | 29.7 | 74.1 | 48.0 | 56.4 | 62.8 |
| paragraph | BM25 | 21.9 | 60.1 | 37.7 | 40.4 | 47.0 |
| paragraph | Dense (MiniLM) | 23.2 | 60.9 | 39.0 | 42.1 | 47.0 |
| paragraph | Hybrid (RRF) | 26.9 | 66.0 | 43.2 | 46.1 | 50.9 |
| paragraph | Hybrid + cross-encoder | 31.6 | 71.7 | 48.3 | 50.9 | 56.9 |
| section | BM25 | 23.1 | 63.5 | 39.5 | 42.8 | 51.8 |
| section | Dense (MiniLM) | 25.6 | 67.1 | 42.5 | 45.6 | 55.1 |
| section | Hybrid (RRF) | 29.3 | 69.7 | 46.4 | 49.5 | 56.9 |
| section | Hybrid + cross-encoder | 32.8 | 75.9 | 50.5 | 53.1 | 63.2 |
| section+header | BM25 | 25.6 | 66.8 | 41.8 | 45.1 | 54.9 |
| section+header | Dense (MiniLM) | 28.7 | 70.0 | 46.0 | 48.2 | 58.8 |
| section+header | Hybrid (RRF) | 32.0 | 72.2 | 49.2 | 52.0 | 60.3 |
| section+header | Hybrid + cross-encoder | 32.8 | 75.9 | 50.6 | 53.3 | 63.3 |

## Search across all 281 papers

| Chunking | Retriever | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Evidence recall@5 |
|---|---|---|---|---|---|---|
| fixed-200w | BM25 | 8.8 | 23.6 | 15.1 | 16.3 | 18.6 |
| fixed-200w | Dense (MiniLM) | 6.6 | 16.1 | 11.0 | 11.1 | 12.3 |
| fixed-200w | Hybrid (RRF) | 10.4 | 23.3 | 15.8 | 16.1 | 17.3 |
| fixed-200w | Hybrid + cross-encoder | 10.5 | 27.5 | 17.3 | 17.9 | 21.6 |
| paragraph | BM25 | 6.8 | 19.6 | 12.6 | 12.8 | 14.0 |
| paragraph | Dense (MiniLM) | 5.0 | 15.1 | 9.4 | 9.5 | 10.5 |
| paragraph | Hybrid (RRF) | 8.1 | 21.2 | 13.6 | 13.3 | 14.9 |
| paragraph | Hybrid + cross-encoder | 11.4 | 27.8 | 17.9 | 17.3 | 20.1 |
| section | BM25 | 8.0 | 22.2 | 13.9 | 14.0 | 16.9 |
| section | Dense (MiniLM) | 6.9 | 16.9 | 11.0 | 10.4 | 12.6 |
| section | Hybrid (RRF) | 9.3 | 21.5 | 15.0 | 14.4 | 16.0 |
| section | Hybrid + cross-encoder | 11.7 | 27.3 | 18.2 | 17.4 | 20.9 |
| section+header | BM25 | 9.1 | 24.6 | 15.3 | 15.7 | 19.1 |
| section+header | Dense (MiniLM) | 7.1 | 15.7 | 10.9 | 10.8 | 11.8 |
| section+header | Hybrid (RRF) | 9.9 | 24.1 | 15.8 | 15.9 | 18.5 |
| section+header | Hybrid + cross-encoder | 11.9 | 28.1 | 18.8 | 18.1 | 21.7 |

## Rerank depth: quality vs latency (section+header chunks)

| Scope | Candidates reranked | Hit@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| in-doc | 5 | 72.2 | 49.6 | 52.3 | 94.0 | 124.1 |
| corpus | 5 | 24.1 | 18.2 | 17.2 | 93.0 | 126.0 |
| in-doc | 10 | 76.7 | 50.5 | 53.0 | 174.6 | 216.2 |
| corpus | 10 | 28.5 | 18.4 | 17.5 | 174.3 | 221.6 |
| in-doc | 20 | 75.9 | 50.6 | 53.3 | 350.2 | 416.6 |
| corpus | 20 | 28.1 | 18.8 | 18.1 | 354.1 | 416.7 |
| in-doc | 30 | 75.6 | 50.5 | 53.2 | 522.8 | 635.8 |
| corpus | 30 | 28.8 | 19.3 | 18.7 | 566.1 | 642.1 |

## End-to-end query latency (200 queries, 7953 chunks indexed, cross-encoder reranks top 10, one query at a time, embedding the query included)

| Retriever | Scope | p50 ms | p95 ms |
|---|---|---|---|
| BM25 | in-doc | 1.2 | 1.5 |
| BM25 | corpus | 1.1 | 1.3 |
| Dense (MiniLM) | in-doc | 5.8 | 7.5 |
| Dense (MiniLM) | corpus | 6.2 | 9.1 |
| Hybrid (RRF) | in-doc | 7.9 | 10.2 |
| Hybrid (RRF) | corpus | 8.4 | 13.2 |
| Hybrid + cross-encoder | in-doc | 191.0 | 232.9 |
| Hybrid + cross-encoder | corpus | 190.4 | 233.8 |

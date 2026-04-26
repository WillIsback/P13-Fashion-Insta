---
gallery_items: 12861
gallery_dims: 768
gallery_model: Marqo FashionSigLIP
n_queries: 500
query_types: non-front views (side, back, additional)
dataset: DeepFashion InShop
k_values: [1, 3, 5, 10]
seed: 42
errors: 0
---

# Recall@K Evaluation Report

## About this metric

**Recall@K** measures the ability of a retrieval system to find at least one relevant item
within the top K results returned for a given query. It is the fraction of queries for which
the correct item appears among the top-K matches.

- **Recall@1 = 80%** → the correct item is the top-1 result in 80% of queries.
- **Recall@5 = 95%** → the correct item is in the top-5 in 95% of queries.
- A higher Recall@K is better; a value of 1.0 means perfect retrieval.
- K=1 is the most stringent; K=10 is the most lenient.

**What is evaluated here:** non-front views (side, back, additional angles) are used as
queries against a gallery of front-view images. Ground truth is any gallery image
sharing the same item ID (`id_XXXXXXXX`). The system must retrieve a matching item
regardless of the view angle, testing view-invariant retrieval.

## Overall — 500 queries

| Metric     | Value  |
|------------|--------|
| Recall@ 1     | 75.80%  |
| Recall@ 3     | 85.20%  |
| Recall@ 5     | 87.40%  |
| Recall@10     | 91.00%  |

## Per Category

| Category            |   n |     @ 1 |     @ 3 |     @ 5 |     @10 |
| ------------------- | --: | ------: | ------: | ------: | ------: |
| Blouses_Shirts      |  75 |  80.00% |  89.33% |  92.00% |  92.00% |
| Cardigans           |  12 |  75.00% |  83.33% |  83.33% |  91.67% |
| Denim               |   9 |  77.78% | 100.00% | 100.00% | 100.00% |
| Dresses             |  62 |  87.10% |  93.55% |  96.77% |  96.77% |
| Graphic_Tees        |   9 |  44.44% |  44.44% |  55.56% |  55.56% |
| Jackets_Coats       |  15 |  86.67% |  93.33% |  93.33% |  93.33% |
| Jackets_Vests       |   6 | 100.00% | 100.00% | 100.00% | 100.00% |
| Leggings            |   4 |  50.00% |  50.00% |  50.00% | 100.00% |
| Pants               |  35 |  60.00% |  80.00% |  82.86% |  94.29% |
| Rompers_Jumpsuits   |  24 |  87.50% |  91.67% |  91.67% |  91.67% |
| Shirts_Polos        |   3 | 100.00% | 100.00% | 100.00% | 100.00% |
| Shorts              |  44 |  65.91% |  79.55% |  79.55% |  88.64% |
| Skirts              |  27 |  74.07% |  81.48% |  81.48% |  92.59% |
| Sweaters            |  26 |  88.46% |  96.15% |  96.15% | 100.00% |
| Sweatshirts_Hoodies |  13 |  84.62% |  84.62% |  92.31% |  92.31% |
| Tees_Tanks          | 136 |  70.59% |  80.88% |  83.82% |  86.03% |

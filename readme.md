# RAFlow

This repository contains the code for RAFlow, a
retrieval-augmented preference-flow model for cold-start cross-domain
recommendation.

## Repository Layout

- `entry.py`: command-line entry for data preprocessing, training, and
  evaluation.
- `run.py`: data loading, training loop, checkpoint selection, and top-K evaluation.
- `raflow_model.py`: RAFlow backbone.
- `config.json`: default task definitions and hyperparameters.

## Data

Download the Amazon 5-core review files

CDs and Vinyl: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz

Movies and TV: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz

Books: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz

and put them in `data/raw/` with these names:

```text
data/
  raw/
    reviews_Books_5.json.gz
    reviews_CDs_and_Vinyl_5.json.gz
    reviews_Movies_and_TV_5.json.gz
```

Then build the intermediate files and train/test splits:

```bash
python3 entry.py --process_data_mid 1
python3 entry.py --process_data_ready 1
```

## Demo

Train and evaluate RAFlow with the `0.8,0.2` split:

```bash
python3 entry.py \
  --task 2 \
  --ratio "[0.8,0.2]" \
  --base_model raflow \
  --epoch 10 \
  --patience 3 \
  --seed 2020 \
  --gpu 0 \
  --eval_steps 5 \
  --lambda_rating 0.1 \
  --lambda_rank 5.0 \
  --rank_neg_num 50 \
  --rank_hard_pool 250 \
  --pop_score_alpha 0.2 \
  --pop_score_mode target_auto \
  --source_memory_alpha 1.0 \
  --source_memory_mode pair_auto \
  --source_memory_topk 100
```

The command trains the model and prints `NDCG@10`, `HR@10`, `MRR@10`,
`Recall@10`, and the corresponding `@20` metrics during evaluation.

## Hyperparameters

| Hyperparameter | Values considered |
| --- | --- |
| Learning rate | `1e-4`, `5e-4`, `1e-3` |
| Embedding dimension | `32`, `64`, `128` |
| Ranking-loss weight `lambda_rank` | `1.0`, `5.0`, `10.0` |
| Rating-loss weight `lambda_rating` | `0.1`, `0.5`, `1.0` |
| Training negatives `rank_neg_num` | `20`, `50`, `100` |
| Target prior weight `pop_score_alpha` | `0.0`, `0.1`, `0.2`, `0.4`, `0.6` |
| Source-neighbor top-K | `20`, `50`, `100`, `200` |
| Source-neighbor weight | `0.0`, `1.0`; with `pair_auto`, use values in `config.json` |
| Heun evaluation steps `eval_steps` | `1`, `2`, `5`, `10`, `20` |


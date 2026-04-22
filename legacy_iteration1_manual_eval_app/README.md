# Manual Evaluation App

This app adds a fast human-labeling workflow for tweet-level ragebait review.

## NOTE: Deprecated / Exploratory Only

This application was built as a temporary testing tool and is kept in the repository purely for historical reference. Using this app helped demonstrate that the underlying data is too messy and that relying on AI-generated tweets will not work for our goals. 

**Known Limitations:**
* **Single-User Architecture:** The backend supports only one user at a time. Concurrent access will result in double-labeling and allow users to unintentionally overwrite or undo each other's actions.
* **Backend State Issues:** The queuing logic is incomplete and occasionally repeats previously shown tweets.

Because this was strictly a proof-of-concept, development on this web app has been abandoned. Moving forward, we will be using [this pre-existing, hand-labeled dataset](https://arxiv.org/pdf/2008.00525) instead.

## What It Creates

On first backend startup, the app will:

- merge `data/unlabeled/unified_unlabeled_posts.csv`
- merge `data/unlabeled/ai_generated_set.csv`
- write a canonical pool to `data/unlabeled/manual_eval_pool.csv`
- tag every AI row with source `gemma4 ai generated`
- use each AI `niche` as its `author_id` bucket so source and author balancing still works
- persist review state in `manual_eval_app/backend/state/manual_eval.sqlite3`
- export labeled rows to `data/labeled/manual_eval.csv`

`manual_eval.csv` is written with these columns:

- `queue_position`
- `post_id`
- `author_id`
- `created_at`
- `language`
- `source`
- `text`
- `label`
- `labeled_at`

`label` uses `0` for not ragebait and `1` for ragebait.

## Run The Backend

From the repo root:

```bash
./.venv/bin/python -m uvicorn manual_eval_app.backend.app.main:app --reload --port 8000
```

## Run The Frontend

```bash
cd manual_eval_app/frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## Keyboard Shortcuts

- `1`: mark as not ragebait
- `2`: mark as ragebait
- `S`: skip
- `Z`: undo
- `I`: toggle info panel

# AI comment detection

Goal: estimate what share of social media comments are written by AI or bots.
Step one is a detector that tells AI-written short comments apart from human ones.

The background research is in `reports/AI comment detection model plan.md`
(sources and notes in `research_notes/`).

## Training data: real tweets + AI posts written about the same things

The human side is the Kaggle Tweet Sentiment Extraction tweets (`train.xlsx`,
about 27k tweets from 2009). For every tweet, a Claude model writes a new post
on the same subject at a similar length. Each human/AI pair is kept together
in one split, so test scores aren't inflated by near-identical text appearing
in both training and testing.

What the pipeline does to keep the detector from learning shortcuts:

- **Same cleanup on both sides** (`aicd/normalize.py`). The tweet file uses
  backticks and garbled characters for apostrophes, censors swear words as
  `****`, and has leftover pieces of deleted usernames. Left alone, the model
  would learn "backtick = human". Curly quotes, em dashes, emoji, links and
  @mentions are also normalized the same way for both.
- **Varied AI text.** Each row gets a random instruction (new post / reply /
  from a short gist), writing style (default voice, lowercase, slang and
  typos, sarcastic, ...) and persona. All choices are fixed per row, so runs
  are repeatable.
- **A model held out for testing.** One generator (by default
  `claude-haiku-5-5`) only appears in the `test_holdout` split, to measure how
  the detector does on a model it never trained on.
- **Fabricated metadata dropped.** The tweet file's `Time of Tweet`,
  `Age of User` and `Country` columns are fake (they repeat in a cycle by row
  number) and are discarded.

## Setup

```bash
pip install -r requirements.txt
```

Put the tweet file at `data/raw/train.xlsx`. The `data/` folder is
git-ignored.

Create your key file. `.env` is git-ignored, so the key never gets committed:

```bash
cp .env.example .env
```

Then open `.env` and paste your key after `ANTHROPIC_API_KEY=`.

## Run

```bash
python -m aicd.prepare_human
```

Cleans the tweets and assigns the train/val/test split.

```bash
python -m aicd.generate_ai --dry-run
```

Shows the plan and five sample prompts. Makes no API calls and costs nothing.

```bash
python -m aicd.generate_ai --limit 200
```

A small paid trial. Read the output, and use the cost estimate it prints at
the end to decide on the full run.

```bash
python -m aicd.generate_ai
```

The full run. It can be resumed: stop it with Ctrl+C, then run the same
command again to continue.

```bash
python -m aicd.build_dataset
```

Merges everything into `data/processed/dataset.csv`.

```bash
python -m aicd.check_leakage
```

Runs quick baselines and the shortcut check.

`check_leakage` trains two quick models. If **surface features alone** (length,
capitals, punctuation) almost perfectly separate human from AI, the dataset has
a formatting shortcut. Fix it before training a real model. The TF-IDF model is
the baseline a fine-tuned transformer must beat. It also prints its most telling
words, so you can spot artifacts.

### Adding recent human comments (strongly recommended)

The tweets are from 2009, but you'll run the detector on comments from 2026.
Collect a few hundred recent, real human comments (for example from Bluesky or
YouTube) and save them as a CSV with a `text` column. Then rebuild the dataset
with them as a test-only set:

```bash
python -m aicd.build_dataset --extra-human-test data/raw/modern_comments.csv
```

`check_leakage` then reports how many of those real people get wrongly flagged.

### Other model providers

`config/generation.yaml` also has entries for OpenAI, Gemini, and free local
models through [Ollama](https://ollama.com) (Llama, Qwen, Mistral). Generators
from other companies make the detector more robust, because a detector trained
only on Claude text mostly learns Claude's habits. Enable any of them with
`enabled: true`. Exactly one enabled generator should have `holdout: true`.

## Tests

```bash
python -m pytest -q
```

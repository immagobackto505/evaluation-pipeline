# Instruction Dataset Evaluation (LLM)

Evaluate LLM responses to instruction–input pairs and export metrics (chrF++, BLEU, optional BERTScore). The pipeline can also auto-query **Gemini 2.5 Flash** to generate responses in batches, then aggregate and score them. 

---

## Features

* Batch LLM inference with **Gemini 2.5 Flash** (saves every 10 rows to `respond_batch/`)
* Merge all batches → `outputs/responded_dataset.csv`
* Metrics:

  * **chrF++** per-sample (`evaluate` → `chrf`)
  * **BLEU (sentence)** via `sacrebleu`
  * **BLEU (corpus)** via `sacrebleu`
  * **BERTScore (optional)** via `evaluate` → `bertscore`
* Domain summary → `outputs/evaluations_summary.csv`

---

## Repository layout

```
.
├─ notebooks/
│  ├─ evaluation.py          # main script
│  └─ evaluation.ipynb       # tutorial/interactive version
├─ outputs/                  # generated (scores, merged CSVs)
├─ respond_batch/            # intermediate batched responses
├─ requirements.txt
└─ .gitignore
```

---

## Data format

Place a `Thai_Chinese_Dataset.csv` in the repo root with at least these columns:

| column        | description                                  |
| ------------- | -------------------------------------------- |
| `instruction` | instruction text                             |
| `input`       | optional input/context string (can be empty) |
| `ref`         | reference/ground-truth text                  |
| `type`        | filename or tag used for domain grouping     |

> The script expects `Thai_Chinese_Dataset.csv` at the project root and writes outputs under `respond_batch/` and `outputs/`. 

---

## Setup

### 1) Create environment

```bash
# (conda recommended)
conda create -n llm-eval python=3.11 -y
conda activate llm-eval
pip install -r requirements.txt
```

### 2) Configure API key (Gemini)

**Do not hardcode keys.** Set an environment variable instead:

**Windows PowerShell**

```powershell
$env:GOOGLE_API_KEY="YOUR_KEY"
```

**macOS/Linux**

```bash
export GOOGLE_API_KEY="YOUR_KEY"
```

> In code, prefer reading from `os.environ["GOOGLE_API_KEY"]` and pass it to the client. (The provided script currently sets a literal key—replace it with the env-var for safety.) 

---

## Run (script)

```bash
# from repo root
python notebooks/evaluation.py
```

### What happens

1. Loads `Dataset.csv`
2. Calls **Gemini 2.5 Flash** to generate `respond` for each row

   * Saves every 10 rows → `respond_batch/batch_{i}.csv`  
3. Merges batches → `outputs/responded_dataset.csv`
4. Computes metrics:

   * chrF++ per row
   * sentence BLEU per row
   * corpus BLEU (once)
   * optional BERTScore (if `torch`+`bert-score` available)  
5. Writes:

   * `outputs/evaluations.csv`
   * `outputs/evaluations_summary.csv` (domain means/stds using normalized `type`)  

---

## Metrics & libraries

* `evaluate` → `load("chrf")` for **chrF++**, `load("bertscore")` for optional **BERTScore**
* `sacrebleu` → `sentence_bleu` and `corpus_bleu`
* `tqdm` for progress bars
* `pandas` for data I/O and aggregation  

---

## Outputs

* `respond_batch/batch_*.csv` — intermediate batches
* `outputs/responded_dataset.csv` — concatenated responses
* `outputs/evaluations.csv` — row-level metrics
* `outputs/evaluations_summary.csv` — means/stds by derived `domain`  

---

## Customize

* **Batch size**: change the modulus `(index+1) % 10 == 0` to your preferred chunk size.  
* **Model**: swap `model="gemini-2.5-flash"` for another model.
* **Language for BERTScore**: change `lang="th"` to match your hypothesis language.  

---

## Troubleshooting

* `ModuleNotFoundError` for `evaluate`/`sacrebleu`: install in the **same** environment that runs the script (`conda activate llm-eval` → `pip install -r requirements.txt`).
* BERTScore skipped: you need `torch` and `bert-score`. The script catches and prints the reason if it can’t compute it.  
* Line endings warning (LF/CRLF on Windows): safe to ignore or set
  `git config --global core.autocrlf true`.


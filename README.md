# Sanskrit–English Instruction Fine-Tuning

Take-home assignment for the AI/ML Developer role at ImmverseAI — **Option 1: Fine-tune an LLM for Sanskrit–English instruction following**.

## Approach

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned with QLoRA (4-bit NF4, LoRA `r=16, alpha=32`) on a single Colab T4/L4.
- **Data:** [`rahular/itihasa`](https://huggingface.co/datasets/rahular/itihasa) for translation (both directions), plus [`JDhruv14/Bhagavad-Gita_Dataset`](https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset) for explanation/transliteration tasks. ~6,800 training examples, 200 held-out eval examples.

## Results

| Direction | Baseline chrF | Fine-tuned chrF |
|---|---:|---:|
| Sanskrit → English | 24.60 | 25.63 |
| English → Sanskrit | 11.08 | 18.77 |

Full metrics, failure analysis, and a domain-relevant Ayurveda passage test are in [`report.md`](./report.md).

## Repo Contents

```
├── README.md
├── sanskrit_llm.ipynb   # training + evaluation notebook (Colab-ready)
└── report.md            # full write-up
```

## Running It

Open `sanskrit_llm.ipynb` in Colab, set runtime to GPU (T4/L4), run all cells. Training uses a manual PyTorch loop instead of `SFTTrainer` to avoid dependency conflicts — see `report.md` §8.

# Report: Sanskrit–English Instruction Fine-Tuning

## 1. Problem Understanding

The goal of this project was to fine-tune a small open-source language model for Sanskrit↔English translation as a practical example of instruction following. I focused mainly on translation instead of putting equal weight on verse explanation and question-answering because the `itihasa` dataset provided a relatively large and clean set of Sanskrit-English parallel examples. Given the 12-hour time limit, this made translation the most practical place to spend the available training time.

## 2. Dataset Preparation

I combined two datasets so that the training data covered more than just translation and included the other task types mentioned in the brief.

- **Translation data:** I used `rahular/itihasa`, which contains aligned Sanskrit-English verses from the Ramayana and Mahabharata. I removed examples with missing fields and filtered out text shorter than 5 characters or longer than 400 characters. For each verse pair, I created two instruction formats: Sanskrit→English and English→Sanskrit.

- **Explanation and transliteration data:** I also used `JDhruv14/Bhagavad-Gita_Dataset`, which contains 701 verses with the fields `chapter, verse, sanskrit, hindi, english, transliteration`. I initially expected to find a commentary or purport field for a proper "explain this verse" task, but the dataset does not contain one. Because of that, I used its English rendering to create an approximate explanation-style task and separately created a transliteration task that converts Sanskrit in Devanagari into Roman script. This also helped address the transliteration variation mentioned in the brief. Output lengths were capped at 600 characters for explanation examples and 300 characters for transliteration examples to keep training manageable.

- Both datasets were converted using the base model's chat template and then merged and shuffled into a single training set.

- For the translation data, I retained the existing train/evaluation split from `itihasa`, keeping evaluation examples separate from training.

- The final training set contained 6,000 `itihasa` translation examples (3,000 verse pairs × 2 directions) and up to around 800 Gita explanation/transliteration examples. The latter came from up to 400 verses that passed the length filter, with two examples created per verse. The evaluation set contained 200 held-out `itihasa` examples. For runtime reasons, 100 of those were used for the generation-based evaluation described later.

## 3. Why This Base Model

I selected `Qwen/Qwen2.5-1.5B-Instruct` for a few practical reasons. It is ungated, so there was no approval step before downloading it, and it was already instruction-tuned. That made it more suitable for this task than starting with a raw base model and training instruction-following behaviour from scratch.

The model was also small enough to work with QLoRA on a single T4 or L4 GPU within the available time. During tokenizer inspection, I noticed that Sanskrit was split into noticeably more subword tokens per character than English. This is consistent with a tokenizer that is more heavily optimized for English and other high-resource languages. In practice, this made Sanskrit inputs more expensive in terms of tokens and likely contributed to some of the problems seen with longer inputs later in the evaluation.

## 4. Fine-Tuning Approach

The model was fine-tuned using QLoRA. I used 4-bit NF4 quantization and LoRA adapters with `r=16`, `alpha=32`, and `dropout=0.05` on the attention and MLP projection layers.

For training, I used a manual PyTorch loop instead of TRL's `SFTTrainer`. The original plan was to use `SFTTrainer`, but repeated dependency conflicts between `trl`, `transformers`, and `numpy` made the setup unreliable in Colab. A simpler loop using `transformers` and `peft` had fewer dependencies and worked more consistently. I used a cosine learning-rate schedule and gradient accumulation, giving an effective batch size of 16.

The initial plan was to train for 400 steps. However, the loss had already started to plateau around step 100, staying mostly within the 1.42–1.57 range. The run was therefore stopped at roughly step 260 instead of using the full 400 steps. This was mainly a time-management decision rather than a training failure, and the evaluation results still showed a clear improvement over the baseline.

The loss started at about 1.70, dropped to around 1.52 by step 40, and then remained relatively stable around 1.4–1.55 for the rest of the run.

## 5. Hardware Constraints and Optimizations

The target environment was a single T4 (16 GB) or L4 GPU in Google Colab.

Using 4-bit quantization and LoRA kept the number of trainable parameters and overall memory usage low enough to train with a batch size of 4 and gradient accumulation of 4, giving an effective batch size of 16. I used `max_steps` rather than full epochs as the main way of controlling the training time.

Memory was not the main problem during the run. No out-of-memory error occurred with the default batch size and gradient accumulation settings on the T4. The bigger limitation was the time required for each step. At roughly 16–17 seconds per step, the originally planned 800-step run was not realistic within the available window. I therefore reduced `MAX_STEPS` to 400 and eventually stopped at around 260 steps after the loss had plateaued.

## 6. Evaluation Methodology

I evaluated the model in three ways:

- **Automatic evaluation:** BLEU and chrF were calculated separately for Sanskrit→English and English→Sanskrit. The baseline model, with the adapter disabled, and the fine-tuned model were evaluated on the same held-out examples.

- **Qualitative evaluation:** I compared baseline and fine-tuned outputs side by side on a fixed set of evaluation examples.

- **Failure analysis:** I inspected examples with the lowest chrF scores and also checked for possible hallucinations using lexical overlap. Generations with almost no word overlap with either the source or the reference were flagged for closer inspection.

| Direction | Baseline BLEU | Baseline chrF | Fine-tuned BLEU | Fine-tuned chrF |
|---|---:|---:|---:|---:|
| Sanskrit → English | 0.31 | 24.60 | 2.68 | 25.63 |
| English → Sanskrit | 0.03 | 11.08 | 0.31 | 18.77 |

The fine-tuned model improved in both directions and on both metrics. The improvement was especially noticeable for English→Sanskrit: chrF increased by about 69% relative to the baseline, while BLEU increased by roughly 10 times. This was also the direction where the original model performed weakest.

The absolute BLEU scores are still low, so they should not be interpreted as evidence of production-level translation quality. BLEU relies heavily on exact n-gram matches and can be particularly harsh for a morphologically rich language such as Sanskrit. chrF is more tolerant of character-level and inflectional differences, so it gives a more useful view of the improvement in this experiment.

## 7. Failure Cases

The model was trained for about 260 of the 400 planned steps before being stopped after the loss plateaued. The 100-example evaluation subset showed two main types of failure.

### English→Sanskrit was the weaker direction

Most of the worst examples came from English→Sanskrit. Eight of the ten examples with the lowest chrF scores were in this direction, as were nine of the ten examples flagged as possible hallucinations. This matches the baseline scores: English→Sanskrit started with a chrF of 11.08 compared with 24.60 for Sanskrit→English.

Interestingly, English→Sanskrit showed the larger relative improvement after fine-tuning, but it was still the more difficult direction in absolute terms.

### The model learned Sanskrit style faster than exact meaning

The most interesting pattern was that some English→Sanskrit outputs sounded convincing even when they were semantically wrong. The generated text often had the right kind of grammatical structure and an epic Sanskrit style, including plausible sandhi, verse-like cadence, and vocabulary such as "महाबल" and "भारत".

For example, one input described peace, personal benefit, modesty, and following the commands of one's parents. Instead of preserving that meaning, the fine-tuned model produced a fluent verse about strength and destiny. In other words, the model appeared to learn the style and surface patterns of Sanskrit before it reliably learned the content mapping.

This is a classic hallucination pattern. Style is easier for a model to reproduce than precise semantic alignment, especially when the amount of fine-tuning is limited.

### Sanskrit→English errors were generally less severe

The Sanskrit→English failures were more often cases of partial understanding rather than completely unrelated generation. In one example, the model correctly recognized that the passage was spoken by a Brahmin and used the appropriate register, but the actual translated content was still incorrect. This is a less serious failure mode than generating fluent but unrelated Sanskrit in the opposite direction.

### Longer inputs were more difficult

Longer, multi-clause sentences appeared frequently among the worst-performing examples and the hallucination cases. This fits with the earlier tokenizer observation: Sanskrit tends to consume more subword tokens with this tokenizer, and longer inputs therefore give the model more opportunities to lose track of the original meaning. The effect was particularly noticeable in the weaker English→Sanskrit direction.

### Hallucination analysis

The lexical-overlap check flagged 9 of the 10 worst cases as English→Sanskrit generations with an `overlap_ratio` of 0.00, meaning that they shared no words with either the source input or the reference. Together with the qualitative review, this suggests that these were not simply slightly inaccurate translations. They were fluent, Sanskrit-looking outputs that had effectively moved away from the requested content.

That distinction matters for a real application. A clearly poor translation is easier for a user to question, whereas a fluent but incorrect Sanskrit passage can appear trustworthy if there is no reference available for comparison.

### Company-relevant qualitative example

I also tested the model on the passage beginning with **"अथ आयुर्वेदः नाम शास्त्रम्..."**, which was provided separately and is relevant to ImmverseAI's Ayurveda/IKS focus. In this case, the fine-tuned model produced a coherent and largely accurate translation and explanation.

**Translation:**  
"The Vedas are called Ayurveda and Ayurveda is called Ayu. The body is called Ayu; mind is called Ayu; soul is called Ayu; senses are called Ayu... The good or bad effects of actions depend on Ayu."

**Explanation:**  
"The Vedas are called Ayurveda and Ayurveda is called Ayu. Ayu means body; Indra means senses; Satva means mind; Aham means self; Sankya means union; that which is beneficial to one's own welfare is said to be Ayurveda."

This result was noticeably better than many of the worst-case examples. One possible reason is that the passage is more definitional and philosophical in nature, which is closer to the Gita explanation-style examples used during training. Narrative sentences that require very precise content transfer were more difficult for the model.

## 8. Challenges Encountered

A few practical issues took up a significant amount of the project time.

- **Colab dependency conflicts:** This was probably the biggest time sink. `trl`'s `SFTTrainer` led to an incompatible `transformers` version, and attempts to pin compatible versions then caused additional `numpy`, `pandas`, and `pyarrow` binary conflicts. I eventually removed `trl` and switched to a manual PyTorch training loop using only `transformers` and `peft`. This reduced the dependency surface and solved the problem.

- **Legacy `itihasa` dataset format:** The `rahular/itihasa` dataset uses an older loading-script format that current versions of the `datasets` library no longer execute, resulting in `RuntimeError: Dataset scripts are no longer supported`. I worked around this by using Hugging Face's auto-generated Parquet mirror with `revision="refs/convert/parquet"`.

- **Gita dataset schema:** The Gita dataset did not contain the commentary/purport field I had originally expected. Its actual fields are `chapter, verse, sanskrit, hindi, english, transliteration`. Instead of presenting the English rendering as a genuine commentary task, I treated it as an approximate explanation task and added a separate transliteration task.

- **Sanskrit tokenization:** Sanskrit was split into more subword tokens per character than English with the selected tokenizer. This likely increased the difficulty of longer examples and is consistent with the error patterns seen during evaluation.

## 9. What I'd Improve With More Time

There are several directions I would take if the project had a larger time or compute budget:

- Train on the full `itihasa` corpus instead of the current subset and allow the run to continue beyond the approximately 260 steps used here. The loss plateau suggests that simply adding steps may not be enough, so increasing the variety and amount of training data would also be important, especially for English→Sanskrit.

- Focus specifically on the English→Sanskrit hallucination problem. Possible approaches include using a lower learning rate, adding more warmup steps, or oversampling English→Sanskrit examples so that the model sees more examples in its weaker direction.

- Adapt the tokenizer by adding frequently occurring Sanskrit subwords and compounds. Reducing token fragmentation could make longer Sanskrit sequences easier to process.

- Find or create a real commentary-based dataset for verse explanation. The Gita dataset used in this project did not contain commentary, so the explanation task was necessarily approximate.

- Compare LoRA with full fine-tuning and, if the available compute permits, experiment with a slightly larger base model.

- Add a human or native-speaker evaluation stage. BLEU and chrF are useful automated measures, but they are still only proxies for translation quality. Human review would be especially useful for distinguishing a translation that is close but imperfect from one that is fluent but completely fabricated.

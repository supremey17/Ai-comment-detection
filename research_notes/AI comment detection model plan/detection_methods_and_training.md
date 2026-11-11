# Detecting AI-Generated Short Social Media Comments: Methods, Accuracy, Training Recipe, Evaluation (as of Nov 2026)

Scope note: research conducted 2026-11-11. Older results (2020-2024) are flagged as such. Account-level bot detection and scraping are out of scope. Where I could only see an abstract (not the full paper), that is stated.

## Approach families and how they compare (classical, fine-tuned transformers, zero-shot, ensembles, LLM-as-judge; benchmark numbers)

### Takeaway
On social-media text, **supervised fine-tuned detectors clearly beat zero-shot/perplexity methods in-distribution** (MultiSocial, ACL 2025: fine-tuned ~0.95-0.98 AUROC vs ~0.72-0.76 for Binoculars/Fast-DetectGPT), and even a small encoder (mDeBERTa-v3-base) is far ahead of zero-shot. But **zero-shot methods (Binoculars, Fast-DetectGPT) can beat fine-tuned models when both the domain and the generator are unseen**, and they have the best low-FPR behaviour among open methods on RAID. Shared-task winners in 2025 were mostly fine-tuned DeBERTa-v3 / small LLM (Qwen2.5-0.5B, Llama-3.2-1B, Gemma-2-2B) classifiers, often ensembled with linguistic features.

### Cited Findings
**MultiSocial (ACL 2025; 472,097 texts, 22 languages, 5 platforms: Telegram, Twitter, Gab, Discord, WhatsApp; 7 generators)** — the most directly relevant social-media benchmark:
- Overall AUROC / Macro-F1@5%FPR (Table 4): fine-tuned Llama-3-8B 0.9769 / 0.8696; Mistral-7B 0.9768 / 0.8692; Aya-101 0.9731 / 0.8462; Falcon-rw-1B 0.9592 / 0.7810; BLOOMZ-3B 0.9582 / 0.7843; XLM-RoBERTa-large 0.9553 / 0.7840; mDeBERTa-v3-base 0.9544 / 0.7652. Zero-shot statistical: DetectLLM-LRR 0.7464; LLM-Deviation 0.7454; S5 0.7418; Fast-DetectGPT 0.7418 / 0.3605; Binoculars 0.7248 / 0.2815. Off-the-shelf pre-trained detectors were near or below chance (Longformer detector 0.4615; RoBERTa-large-OpenAI-Detector 0.3450 / 0.1376) — [MultiSocial, ACL 2025](https://aclanthology.org/2025.acl-long.36/)
- ">7b parameter models achieved significantly superior performance compared to <7b models", but "even the smallest mDeBERTa-v3-base achieved much better performance than zero-shot detectors (which also use base models of >6b parameters)" — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- "If a different generator is used (i.e., cross-domain and cross-generator), the Fast-Detect-GPT and Binoculars can outperform the fine-tuned detectors" — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- Best zero-shot statistical vs best pre-trained detectors were roughly equal (0.72-0.75 vs 0.72-0.76 AUROC) — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)

**COLING 2025 GenAI Detection Task 1 (English subtask, 36 teams, binary human vs machine):**
- Top 3 macro-F1: Advacheck 83.07, Unibuc-NLP 83.01, Fraunhofer SIT 82.80; RoBERTa baseline 73.42 (test). Only 14 of 36 teams beat the baseline. Dev-set baseline was 96.2 accuracy but dropped to 74.9 on test, showing in-distribution scores massively overstate real performance — [Task 1 overview, arXiv 2501.11012](https://arxiv.org/abs/2501.11012)
- Winner Advacheck: multi-task DeBERTa-v3-base shared encoder with a binary head plus auxiliary domain-classification heads. Unibuc-NLP (2nd): XLM-RoBERTa-base and Qwen2.5-0.5B (Qwen classifier performed on par); LoRA fine-tuning. Fraunhofer SIT (3rd): MGT adapter + NLI adapter over RoBERTa-base. Grape (4th, and 1st multilingual at 79.16 macro-F1): fine-tuned Llama-3.2-1B and Gemma-2-2B, ensembled with linguistic features. TechExperts (5th): fine-tuned Gemma-2B — [Task 1 overview](https://arxiv.org/pdf/2501.11012v2)
- Per-domain test accuracy for the winner included Tweet 94.5 (only 1,325 tweet test items), News 65.1, Summary 70.0 — [Task 1 overview](https://arxiv.org/pdf/2501.11012v2)

**COLING 2025 Task 3 (cross-domain, on RAID):** 9 teams / 23 submissions; multiple participants reported >99% accuracy on RAID machine text at 5% FPR (non-adversarial; I could not verify the adversarial breakdown) — [Task 3 overview, arXiv 2501.08913](https://arxiv.org/abs/2501.08913)

**RAID (ACL 2024 — older; >6M generations, 11 generators, 8 domains incl. Reddit, 11 attacks, 4 decoding strategies; accuracy at 5% FPR):**
- Baseline (no-attack) accuracy for detectors in the attack study: Originality 85.0, Binoculars 79.6, RADAR 70.9, GPTZero 66.5, GLTR 62.6, RoBERTa-large-GPT2 56.7 — [RAID paper](https://arxiv.org/html/2405.07940v2)
- Binoculars "performs particularly well at low FPR"; few detectors can operate below 1% FPR; some plateau (ZeroGPT can't go below 16.9% FPR, Fast-DetectGPT 0.88%, Originality 0.62%) — [RAID paper](https://arxiv.org/html/2405.07940v2)
- Repetition penalty lowers detector accuracy by up to 32 points; sampling is harder to detect than greedy — [RAID paper](https://arxiv.org/html/2405.07940v2)
- Live leaderboard (split into trained-on-RAID vs not) at raid-bench.xyz/leaderboard; dataset on HF `liamdugan/raid`; `raid-bench` pip package evaluates accuracy at a target FPR (default 5%) — [RAID GitHub](https://github.com/liamdugan/raid); I could not retrieve current leaderboard numbers (page returned 404 to my fetch).

**Classical/stylometric:**
- Stylometric features on 463,382 Weibo comments (Chinese, LLM bots vs humans) reached 91.8% F1 for short-comment detection — [CCL 2025, ACL Anthology](https://aclanthology.org/2025.ccl-1.64/)
- TweepFake (2020/21 — old, GPT-2/RNN-era bots): 25,572 balanced tweets; of 13 methods, fine-tuned RoBERTa was best at ~89.6% accuracy, ahead of classical BoW/char-level approaches — [TweepFake, arXiv 2008.00036](https://arxiv.org/abs/2008.00036)

**Newer (2026) probing approach:**
- Linear probes on LLM hidden representations: MGT and human text are "linearly separable in low-dimensional space"; ~+11 AUC out-of-domain improvement over 16 baselines on 4 benchmarks; <100 samples to reach near-peak performance (abstract only; Aug 2026 preprint, not peer reviewed) — [arXiv 2608.24780](https://arxiv.org/abs/2608.24780)

**Zero-shot proxy sensitivity:** Fast-DetectGPT and Binoculars performance is highly sensitive to the choice of proxy/scoring model — [Can We Trust LLM Detectors?, arXiv 2601.15301](https://arxiv.org/html/2601.15301)

### Inferences
- For a solo developer targeting short English comments, the best cost/benefit is a fine-tuned encoder (DeBERTa-v3-base/large or ModernBERT) or a small decoder LLM classifier (Qwen2.5-0.5B-1.5B, Llama-3.2-1B, Gemma-2-2B with LoRA), plus a Binoculars/Fast-DetectGPT score as an extra ensemble feature for robustness to unseen generators.
- Classical TF-IDF/stylometry + LR/XGBoost is a useful fast baseline and ensemble member, but evidence (TweepFake, COLING) shows transformers dominate.
- The gap between dev (96%) and blind test (75%) in COLING 2025 is the single most important lesson: any number measured on a random split of your own data is optimistic.

### Gaps
- I did not retrieve verified numbers for M4/M4GT, MAGE, DetectRL, or SemEval-2024 Task 8 leaderboards; DetectRL-X (ACL 2026) abstract confirms length/attack/generator dimensions but I could not see its result tables — [DetectRL-X, arXiv 2605.15518](https://arxiv.org/abs/2605.15518).
- No verified head-to-head of ModernBERT vs DeBERTa-v3 for AI-text detection was found.
- No quantitative LLM-as-judge results (e.g., prompting GPT/Claude to classify) were retrieved; COLING Task 1 mentions one team exploring "LLaMA-as-a-judge" but not with top results.
- Current RAID leaderboard rankings (late 2026) not retrieved.

## Short texts and social-media datasets

### Takeaway
Detection degrades on short text, most severely for zero-shot and commercial tools below ~50 words; but fine-tuned models trained **on short social-media text itself** still reach ~0.95+ AUROC in-distribution (MultiSocial median lengths only 9-18 words). The biggest short-text threat is not length per se but **fine-tuned/private generators and human-mimicking prompts**, which cause detectability to "drop dramatically".

### Cited Findings
- Chicago Booth/BFI study (Jabarian & Imas, Sept 2025; ~2,000 human passages, 6 genres, 4 LLMs, length bands from <50 words to ~1,000): all three commercial tools (GPTZero, Originality, Pangram) lost accuracy on very short passages (<50 words); open-source RoBERTa was often near random and "unsuitable for high-stakes applications" — [Chicago Booth Review](https://www.chicagobooth.edu/review/2025/december/do-ai-detectors-work-well-enough-trust); [BFI WP 2025-116](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf)
- MultiSocial median word-counts: Discord 9 words, WhatsApp/Twitter 18; length did not clearly explain platform differences for zero-shot methods; fine-tuned detectors were slightly weaker on generators with shorter outputs (OPT-IML-Max-30B, Aya-101) — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- M4 (2023, older): truncating text from 1,000 to 125 characters cut ChatGPT-text detection accuracy by 4.5 points — [M4, arXiv 2305.14902](https://arxiv.org/abs/2305.14902)
- Survey: shrinking input from 256 to 64 words cost ~10% accuracy for nucleus-sampled text but <1% for top-k text — [arXiv 2406.15583](https://arxiv.org/abs/2406.15583)
- LLM-DetectAIve restricts inputs to 50-500 words because detectors are much less accurate on shorter text — [arXiv 2408.04284](https://arxiv.org/abs/2408.04284)
- Short-PHD (2025) targets ~50-token texts with a topological zero-shot method and reports results at 50/100/150 tokens (numbers not retrieved) — [arXiv 2504.02873](https://arxiv.org/abs/2504.02873)
- "When Detection Fails" (Dawkins, Fraser, Kiritchenko, 2025): 505,159 AI social-media posts on 11 controversial topics from open, closed and fine-tuned LLMs; posts are detectable when the detector knows/has access to the generator, but when attackers keep a fine-tuned model private, "detectability drops dramatically" (confirmed with a human study) — [arXiv 2506.09975](https://arxiv.org/abs/2506.09975)
- COLING 2025 Task 1: detectors showed "remarkable vulnerability when machine-generated text is adapted to mimic humans, either by introducing typo, link, and tags, or by using fill-human-machine gap prompts" — [arXiv 2501.11012](https://arxiv.org/pdf/2501.11012v2)

**Public datasets useful for training/eval (social/short-text focus):**
| Dataset | Content / size | Link |
|---|---|---|
| MultiSocial (ACL 2025) | 472,097 texts; 22 languages; Telegram, Twitter, Gab, Discord, WhatsApp; 7 generators (incl. Aya-101, Mistral-7B-Instruct, Gemini, OPT-IML) | [Zenodo 10.5281/zenodo.13846152](https://doi.org/10.5281/zenodo.13846152); [code](https://github.com/kinit-sk/multisocial) |
| TweepFake (2020, old generators) | 25,572 tweets, balanced human/bot | [arXiv 2008.00036](https://arxiv.org/abs/2008.00036) |
| AIGTBench (ACL 2025) | Medium, Quora, Reddit; ~2.4M collected posts + AI texts from 12 LLMs; best detector OSM-Det | [arXiv 2412.18148](https://arxiv.org/abs/2412.18148) |
| RAID | >10M docs incl. Reddit domain, 11 generators, 11 attacks; train split 802M (11.8G with attacks) | [HF liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid) |
| COLING 2025 Task 1 | English training set aggregated from many sources (incl. tweets, CUDRT, IELTS, PeerReview, MixSet) | [GitHub mbzuai-nlp/COLING-2025-Workshop-on-MGT-Detection-Task1](https://github.com/mbzuai-nlp/COLING-2025-Workshop-on-MGT-Detection-Task1) |
| "When Detection Fails" | 505,159 AI social posts (release status not confirmed) | [arXiv 2506.09975](https://arxiv.org/abs/2506.09975) |
| Weibo comments (Chinese) | 463,382 comments | [CCL 2025](https://aclanthology.org/2025.ccl-1.64/) |
| RedNote-Vibe | Xiaohongshu longitudinal data, pre-LLM to July 2025 | [aitopics record](https://aitopics.org/doc/arxivorg:DB1B0AC8) |
| reddit-ai-detection (HF) | small: ~2,300 human Reddit texts, only 79 from 2022+ | [HF](https://huggingface.co/datasets/validname/reddit-ai-detection/blob/main/README.md) |
| DetectRL-X (ACL 2026) | 8 languages, 6 domains, 4 commercial LLMs, polish/expand/condense ops, attacks | [GitHub](https://github.com/AIDC-AI/Marco-LLM/tree/main/DetectRL-X) |

- Real-world prevalence: a study of 51 subreddits (2022-2024) found synthetic text "marginally present", with some communities peaking at ~9% in some months (conservative estimate, concentrated in few users) — [Machines in the Crowd?, arXiv 2510.07226](https://arxiv.org/abs/2510.07226). This informs realistic class priors (see Evaluation).

### Inferences
- Pre-2019 (pre-GPT-2) Reddit/YouTube/Twitter comments are a clean, cheap source of guaranteed-human negatives (this is how the ICWSM 2026 audit defines ground truth; see Robustness section).
- Since MultiSocial shows ~10-20-word texts are learnable when training data matches, the key is to generate in-platform AI comments (same thread context, length, slang, emoji) rather than rely on essay/news datasets.

### Gaps
- No public YouTube AI-comment benchmark with generated positives was confirmed; I found no 2026 dataset covering modern generators (GPT-5, Claude 4.x, Gemini 2.5/3) on short comments.
- No precise accuracy-vs-word-count curve under 50 words for modern detectors was retrieved (Chicago Booth reports drop qualitatively in the article).

## Robustness: unseen generators, paraphrase/humanizers, domain shift, non-native bias

### Takeaway
Every detector family has a known break: homoglyph/character attacks crush many detectors (Originality 85.0 to 9.3 on RAID), synonym swaps hurt perplexity methods (Binoculars -36 pts), DIPPER paraphrasing broke DetectGPT (70.3% to 4.6% at 1% FPR), and supervised models lose much of their edge on unseen domains/platforms/generators. False positives against non-native writers, autistic writers, and specific political communities are documented, so threshold at very low FPR and audit on real pre-LLM human text.

### Cited Findings
- RAID attack table (accuracy at 5% FPR, drop from baseline): Homoglyph: Originality 9.3 (-75.7), Binoculars 37.7 (-41.9), GLTR 24.3, RoBERTa-L-GPT2 21.3, RADAR 59.3, GPTZero 66.2 (-0.3). Synonym: Binoculars 43.5 (-36.1), GLTR 31.2 (-31.4). Paraphrase: Binoculars 80.3 (+0.7), Originality 96.7 (+11.7). Adversarially trained RADAR was more robust overall — [RAID](https://arxiv.org/html/2405.07940v2)
- DIPPER (11B paraphraser, NeurIPS 2023, older): DetectGPT accuracy 70.3% to 4.6% at 1% FPR; also evaded watermarking, GPTZero, OpenAI classifier; retrieval defense detects 80-97% of paraphrases at 1% FPR (only for API providers) — [Krishna et al.](https://arxiv.org/abs/2303.13408)
- Cross-platform: training on Discord gives best Discord performance but least transferability (AUROC drop of 27% for Llama-3-8B on other platforms); on Twitter test data Discord-trained detectors were 13% lower AUROC than Telegram-trained; Telegram (diverse lengths/forms) or multi-platform training transfers best; Gab (toxic content) hardest — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- Unseen generator: MultiSocial held out Gemini from training and saw slightly decreased performance on it; cross-domain + cross-generator, zero-shot beat fine-tuned — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- COLING 2025 Task 1: all systems good on seen domains but "significant declines on unseen domains and languages" — [arXiv 2501.11012](https://arxiv.org/pdf/2501.11012v2)
- Fine-tuned private generators make social posts much less detectable — [arXiv 2506.09975](https://arxiv.org/abs/2506.09975)
- ICWSM 2026 social web audit (Dutta et al.): using Reddit/YouTube posts from before 1 Jan 2019 as ground-truth human, SoTA detectors "exhibit considerable false positives" and flag liberal political discourse as AI more often than conservative discourse — [A Large Scale Social Web Audit of AI Generated Text Detection Systems, ICWSM 2026](https://ojs.aaai.org/index.php/ICWSM/article/view/42660) (full text blocked to my fetch; numbers not retrieved)
- Non-native bias (2023, older): Liang et al. found GPT detectors consistently misclassify non-native English writing (TOEFL essays) as AI — widely reported as ~61% average FPR vs ~5% for native essays (figure seen via secondary summaries) — [Liang et al., arXiv 2304.02819](https://arxiv.org/abs/2304.02819). 2026 preprint argues professional editing style is a confound (135,389 document pairs) — [arXiv 2608.26710](https://arxiv.org/pdf/2608.26710)
- Pangram reports 0% FPR on 91 TOEFL essays, 0 FPs on ELLIPSE (3,907), 0.09% on ICNALE (5,600), vs GPTZero 7.7% on TOEFL (vendor-authored) — [Pangram tech report](https://arxiv.org/html/2402.14873v3)
- Misclassification of autistic writing as AI is the subject of a 2026 preprint (findings not read) — [arXiv 2607.14729](https://arxiv.org/pdf/2607.14729)

### Inferences
- Normalize Unicode (NFKC, homoglyph mapping, zero-width/whitespace stripping) before inference — this cheaply neutralizes the single most damaging RAID attack.
- Augment training with paraphrased (e.g., DIPPER-style / LLM rewrite), typo-injected, synonym-swapped and "humanized" AI comments; RADAR's robustness in RAID supports adversarial training.
- Include ESL/non-native, informal, and politically diverse human comments as hard negatives and report FPR per subgroup.

### Gaps
- No 2025-2026 quantitative benchmark of commercial "humanizer" tools (StealthGPT, Undetectable.ai, etc.) against short-comment detectors was found.
- No verified per-generator numbers for GPT-4o/GPT-5/Claude/Gemini on short comments.

## Training recipe

### Takeaway
Best documented practice: fine-tune a strong encoder (DeBERTa-v3-base, as the COLING 2025 winner) or a 0.5-8B decoder with LoRA/QLoRA, on a domain-matched mixture of many generators; generate "mirror" AI comments that match each human comment's topic/length/context; then iterate **hard-negative mining** (add the human texts the model flags) — Pangram's report shows this cut FPR from 2.29% to 0.02%. Calibrate thresholds on held-out data at a fixed low FPR.

### Cited Findings
- Pangram recipe: ~28M pre-2022 human documents across 10 domains (4M held out for FPR eval); each human document gets an LLM "synthetic mirror" matching topic and length (some prompts include the original's opening to steer style; domain-tuned prompts; post-filtering of AI boilerplate) — [Pangram tech report](https://arxiv.org/html/2402.14873v3)
- Pangram scaling: performance saturated around 40k examples per domain (~400k total); 100k per domain still did not exceed 98% accuracy without mining — [Pangram](https://arxiv.org/html/2402.14873v3)
- Pangram hard negative mining: start 40k/domain, each round add up to 10k false positives per domain plus their mirrors, retrain; domain-weighted FPR 2.29% to 0.02% (email 6.60% to 0.00%, Wikipedia 5.34% to 0.05%) — [Pangram](https://arxiv.org/html/2402.14873v3)
- COLING winner: DeBERTa-v3-base with auxiliary domain-classification heads (multi-task) — [arXiv 2501.11012](https://arxiv.org/pdf/2501.11012v2)
- LoRA fine-tuning of XLM-R / Qwen2.5-0.5B performed strongly (2nd place) — [arXiv 2501.11012](https://arxiv.org/pdf/2501.11012v2)
- MultiSocial fine-tuning hyperparameters: QLoRA, learning rate 2e-4, batch size 2 with gradient accumulation 8 (effective 16); cross-lingual runs extended to 7 epochs — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- Multi-platform training gives the best average cross-platform transfer vs single-platform (Telegram as exception) — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- Repetition penalty and sampling settings materially change detectability (up to 32 pts) — so generate training data with varied decoding — [RAID](https://arxiv.org/html/2405.07940v2)
- Linear probes on LLM hidden states reach near-peak performance with <100 labelled samples (2026 preprint, abstract only) — [arXiv 2608.24780](https://arxiv.org/abs/2608.24780)

### Inferences (recommended recipe for a solo developer; my synthesis, not a single source)
1. **Model**: start with `microsoft/deberta-v3-base` (or ModernBERT-base for speed), max_len 128 tokens (comments are short; MultiSocial medians 9-18 words), lr ~2e-5, batch 32, 2-4 epochs, warmup 6-10%, weight decay 0.01. If GPU allows, a LoRA-tuned Qwen2.5-1.5B / Llama-3.2-1B / Gemma-2-2B classifier (QLoRA lr 2e-4, effective batch 16 as in MultiSocial) is the next step up.
2. **Data**: human negatives from pre-2022 (ideally pre-2019) comments on the target platform plus recent human comments; AI positives generated by >=8-10 diverse generators (GPT-4o/5-class, Claude, Gemini, Llama, Qwen, Mistral, DeepSeek, small local models), each prompted with the actual post/thread context and a target length drawn from the human length distribution, with varied temperature/top-p/repetition penalty and persona/style instructions ("write like a casual redditor, lowercase, typos, emoji").
3. **Augmentation**: paraphrase/humanizer passes, typo/case/emoji/slang injection, truncation, homoglyph and whitespace noise on both classes (so noise is not a class cue).
4. **Hard negatives**: iterate Pangram-style mining over a large unlabeled pool of guaranteed-human comments; add flagged humans + mirrors each round.
5. **Ensemble**: average or stack the fine-tuned probability with a Binoculars/Fast-DetectGPT score and a few stylometric features (helps on unseen generators per MultiSocial).
6. **Calibration**: temperature scaling on a held-out calibration split; choose threshold for target FPR (e.g., 1% or 0.5%) on held-out human data from the target platform, not on training data.

### Gaps
- No public ablation found on optimal max sequence length or emoji/typo augmentation specifically for short comments.
- No verified published benchmark comparing ModernBERT vs DeBERTa-v3 for this task.

## Evaluation methodology

### Takeaway
Report AUROC plus TPR at fixed low FPR (1% and 5%), with thresholds set on separate calibration data; evaluate on held-out generators, held-out platforms/domains, attacked text, and real pre-LLM human comments; use realistic low prevalence (Reddit ~marginal, up to ~9% in hot spots) to estimate precision.

### Cited Findings
- RAID standard: accuracy (TPR) at 5% FPR with per-domain thresholds recommended; a single global threshold gives Reddit FPRs of 3.7% (RoBERTa-B-GPT2), 10.7% (RADAR), 7.2% (Binoculars), 1.1% (GLTR) — [RAID](https://arxiv.org/html/2405.07940v2)
- MultiSocial uses AUROC plus Macro-F1@5%FPR with threshold calibrated on train-data ROC "to avoid data leakage", and holds out a generator (Gemini) for unseen-generator evaluation — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- Pangram and the Booth study set each detector's threshold at a fixed FPR (1%) and compare recall; Booth proposes "policy caps" such as <=0.5% human text flagged — [Pangram](https://arxiv.org/html/2402.14873v3); [Chicago Booth Review](https://www.chicagobooth.edu/review/2025/december/do-ai-detectors-work-well-enough-trust)
- COLING 2025 Task 1 dev 96.2 vs test 74.9 accuracy for the baseline; organisers note possible near-duplicates across aggregated sources and that pretrained models may have seen test data — [arXiv 2501.11012](https://arxiv.org/pdf/2501.11012v2)
- ICWSM 2026 audit uses pre-2019 social posts as near-certain human ground truth to measure real FPR — [ICWSM 2026](https://ojs.aaai.org/index.php/ICWSM/article/view/42660)
- Prevalence on Reddit: marginal overall, up to ~9% in some subreddits/months — [arXiv 2510.07226](https://arxiv.org/abs/2510.07226)

### Inferences
- Leakage controls: split by thread/post (AI comment and its mirrored human comment, and all replies to one post, stay in the same split); dedupe near-duplicates (MinHash); split by author for humans; hold out entire generators and entire platforms/subreddits; time-based split (train older, test newer).
- At 2% prevalence, a detector with 90% TPR at 1% FPR yields precision ~65%; at 5% FPR precision falls to ~27% — which is why low-FPR operating points matter (arithmetic, not sourced).
- Report bootstrap confidence intervals and per-subgroup FPR (length buckets <10, 10-25, 25-50 words; ESL; platform).

### Gaps
- No standard 2026 short-comment leaderboard found to benchmark against.

## Commercial / open detectors to benchmark against

### Takeaway
Independent 2025 evaluations rank Pangram best (near-zero FPR, ~99.8-100% ranking accuracy), GPTZero close, Originality higher FPR/FNR; all degrade under 50 words. Off-the-shelf open RoBERTa detectors (OpenAI GPT-2 detector, etc.) perform near/below chance on modern social text and are only useful as weak baselines.

### Cited Findings
- Booth/BFI (Sept 2025): FPR <1% for GPTZero, Originality, Pangram (Pangram ~0 at most thresholds); FNR GPTZero ~0-2%, Pangram ~2-4%, Originality ~10-40% depending on model; Pangram ranking accuracy never below 99.8%; GPTZero 96% even on short passages; Pangram only detector meeting policy-grade caps across all 4 LLMs — [Chicago Booth Review](https://www.chicagobooth.edu/review/2025/december/do-ai-detectors-work-well-enough-trust)
- Epoch AI: on 495 human passages Pangram and GPTZero had zero false alarms; Originality flagged 19 (3.8% FPR); detectors missed more on author-style-mimicking AI text — [Epoch AI data insight](https://epoch.ai/data-insights/ai-text-detection-challenges)
- RAID (2024): Originality 85.0, GPTZero 66.5 accuracy at 5% FPR; GPTZero robust to homoglyphs, Originality not — [RAID](https://arxiv.org/html/2405.07940v2)
- Pangram's own report: 99% accuracy on 1,976-doc benchmark; GPTZero FNR 10.02%, Originality FPR 9.24% (vendor-authored, older version) — [Pangram](https://arxiv.org/html/2402.14873v3)
- Vendor claims (unverified): GPTZero 0.24% FPR on its own benchmark; Pangram "99.98% accuracy" — reported via search summaries; treat as marketing.
- Open HF baselines: RoBERTa-large-OpenAI-Detector 0.345 AUROC and Longformer detector 0.46 on MultiSocial; fine-tuned MultiSocial checkpoints (e.g., Llama-3-8b-MultiSocial, mDeBERTa-v3-base-MultiSocial) are the strongest open social-media baselines identified — [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)
- Open zero-shot: Binoculars ([arXiv 2401.12070](https://arxiv.org/abs/2401.12070)) and Fast-DetectGPT are the standard training-free baselines.

### Inferences
- Benchmark plan: Pangram and GPTZero APIs (if budget allows), Binoculars (Falcon-7B pair), Fast-DetectGPT, a TF-IDF+LR baseline, and a MultiSocial fine-tuned checkpoint, all scored at the same FPR thresholds on your held-out comment test set.

### Gaps
- No independent 2026 numbers for commercial detectors on <50-word social comments specifically.
- Specific open HF detectors popular in 2025-2026 (e.g., Desklib, SuperAnnotate, Fakespot) were not verified in this session.

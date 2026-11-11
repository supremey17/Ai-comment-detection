# Replace your three datasets before training anything

None of the three uploaded datasets can train a usable AI-comment detector, and two of them would actively mislead you. **archive_3 is random word salad** drawn from a 971-word vocabulary, and its labels carry no signal: a TF-IDF classifier scores **exactly 0.500 accuracy**. **archive_5 is 2,000 rows of fill-in-the-blank templates** with only 966 unique texts. A model scores 1.000 on it by memorizing templates, then **falls to 0.359 on templates it has not seen**. **archive_4 is real 2009-era human tweets** labelled for sentiment, with no AI label at all. Models trained on archive_3 and archive_5 flagged **32.5% and 17.5% of genuine human tweets as AI**. That would invent a bot epidemic from noise, because credible estimates put AI-written comments in the low single digits. The workable path has five parts:

1. Keep archive_4 only as human negatives, and add pre-LLM human comments and public social-media detection corpora.
2. Generate your own AI comments from 8–10 current models, each matched to a real thread.
3. Fine-tune a DeBERTa-v3-class encoder and judge it at a 1% false-positive rate on generators it never saw.
4. Collect target comments from Bluesky and YouTube through official APIs.
5. Turn detector flags into a prevalence estimate using a hand-labelled validation sample and adjusted classify-and-count, not the raw flag rate.

The detector is only half the project. The other half is a measurement design that corrects for the detector's errors, and that is what makes a headline percentage defensible.

## Two uploads are noise and the third has no AI label

The audit measured each file directly with pandas and scikit-learn. The scripts are in `/tmp/claude-0/-home-user-Ai-comment-detection/487b0154-22ad-5df6-93c4-8475ac181bd5/scratchpad/scripts/`, and the figures in this section come from them unless a link is given. The baseline was one fixed setup throughout: word and character TF-IDF with logistic regression, scored by stratified 5-fold cross-validation.

**archive_3 (`ai_human_content_detection_dataset.csv`) should be discarded.** It has 1,367 rows split 684/683 across eight content types, plus 14 precomputed "linguistic" features. The text is not language. It uses only **971 distinct word types across 191,641 tokens**, with a nearly flat frequency distribution, and has no question marks, apostrophes, digits, emoji or URLs in either class. That pattern points to a lorem-ipsum generator such as Python Faker rather than human or LLM writing. Samples read like "Fly speech car hope picture make wear." The two classes are statistically identical (mean Flesch score 52.18 in both), and every model the audit tried landed at chance. TF-IDF scored 0.500 accuracy, gradient boosting on the 14 features scored 0.520, and the best single feature (burstiness) reached an AUC of only 0.543. The file matches a Kaggle dataset that describes itself as synthetic ([Kaggle](https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025)). A 2025 paper appears to report results on a dataset of the same 1,367×17 shape ([Atlantis Press](https://www.atlantis-press.com/article/126024937.pdf)), though the audit could not confirm it is the same file. If it is, those published accuracies measure noise. The file also never says whether label 1 means AI.

**archive_5 (`ai_vs_human_text_2026.csv`) is templated, so a model learns the templates instead of AI style.** Its Kaggle description says it holds GPT-4o and Gemini 2.0 outputs from "style simulation" prompts across academic, news and social domains ([Kaggle](https://www.kaggle.com/datasets/alitaqishah/ai-vs-human-text-classification-dataset-2026)). In practice the file has several problems:

- **Duplicates:** 1,034 of 2,000 rows are exact duplicates, and one "human" text appears 11 times.
- **Templates:** if you replace the topic phrase with a placeholder, only **18 AI templates and 29 human templates** remain.
- **Model names:** each AI template belongs to exactly one model, so `source_model` carries no information beyond the template.
- **Leaky columns:** `generation_method` reproduces the label exactly.
- **Shallow "human variation":** human social posts are 100% lowercase-initial and AI ones never are. Emoji (only 🧵, 👇 and 😭) and ALL-CAPS words appear only in human rows. The only non-ASCII character in AI rows is the em dash.
- **Human text that is not human:** the "human" academic and news texts read like LLM prose, and some are ungrammatical where a template slot was filled badly ("Officials confirmed on Thursday that aging population challenges.").

The result is **1.000 accuracy on random splits**. When whole templates are held out, accuracy collapses to **0.359**, which is below the 0.667 you would get by always guessing "human". A detector trained on this file would learn that lowercase-plus-slang means human and polished means AI. Real AI comments prompted to sound casual defeat that rule easily.

**archive_4 (`train.xlsx`) is the only real social text, and it has no AI label.** It contains 27,481 English tweets from the Kaggle Tweet Sentiment Extraction competition, sourced from Figure Eight's Data for Everyone platform ([cengc13](https://cengc13.github.io/website/kaggle/2020/06/01/kaggle-tweet-start-blog.html); [arXiv 2104.09777](https://arxiv.org/pdf/2104.09777)). The labels are neutral, positive and negative. The `Time of Tweet` and `Age of User` columns cycle exactly with the row index, and `Country` is spread almost evenly over 195 countries. Those three columns are fabricated and must never become features. The text also has collection artifacts: 8,807 rows use a backtick as the apostrophe, 10,975 start with a leading space, and none contain emoji. Left as they are, a model learns that a backtick means human.

| File | Rows | What it really is | Signal for AI detection | Verdict |
|---|---|---|---|---|
| archive_3 | 1,367 | Faker-style word salad, random labels | None (CV accuracy 0.500) | Discard |
| archive_5 | 2,000 (966 unique) | 47 templates × 30 topics | Template identity only (0.359 on unseen templates) | At most a tiny sanity check, with metadata dropped |
| archive_4 | 27,481 | Real 2009-era human tweets | Human negatives only | Keep after normalizing; drop metadata |

The cross-dataset tests show how much damage these files would do in production. A model trained on archive_3 and tested on archive_5 reached an AUC of 0.578, and the reverse reached 0.529. Both are close to chance. More importantly, both models flagged large shares of the archive_4 tweets, which are known to be human: **32.5%** for the archive_3 model and **17.5%** for the archive_5 model. Point either model at a real comment section and nearly everything it flags would be a false positive. None of the files covers short AI comments, any 2025–2026 generator, paraphrased or "humanized" text, or modern emoji-heavy comment style.

## Build training data that mirrors the comments you will score

The most important design choice is to make the AI and human examples differ only in authorship. Topic, length, platform and era should be the same on both sides. Pangram's detector does this with "synthetic mirrors": each human document is paired with an LLM-generated text matched on topic and length, and some prompts include the original's opening lines to steer the style ([Pangram](https://arxiv.org/html/2402.14873v3)). Copy that design.

**Human side.** Three sources work together:

- **archive_4, cleaned:** about 27,200 unique tweets after normalizing (backticks to apostrophes, whitespace stripped, near-duplicates removed, metadata columns dropped).
- **Comments written before 2019, which are almost certainly human.** The ICWSM 2026 social-web audit used Reddit and YouTube posts from before 1 January 2019 as its human ground truth ([ICWSM 2026](https://ojs.aaai.org/index.php/ICWSM/article/view/42660)). You can collect old YouTube comments legally through the official API (see the collection section).
- **Recent human comments** from your target platforms, so the model does not learn that "2009 slang means human."

The era confound is real. If every human example is old and every AI example is new, the model learns to detect the date, not the author. Mirroring each AI comment to the same thread as a human comment is what neutralizes that.

**AI side.** Generate comments yourself, because no public dataset covers modern models writing short comments.

- **Use many models:** spread generation across 8–10 generators (GPT-4o/5-class, Claude, Gemini, Llama, Qwen, Mistral, DeepSeek and a small local model).
- **Give real context:** prompt each one with the actual post or thread a human comment replied to, and draw the target length from the human length distribution.
- **Vary decoding:** RAID found that repetition penalty and sampling settings alone move detector accuracy by **up to 32 points** ([RAID](https://arxiv.org/html/2405.07940v2)).
- **Include imitation prompts:** prompt some comments to imitate people ("lowercase, a typo, one emoji, sound like a redditor"). At COLING 2025, detectors showed "remarkable vulnerability" to AI text adapted to mimic humans with typos, links and tags ([COLING 2025 Task 1](https://arxiv.org/pdf/2501.11012v2)). A 2025 study of 505,159 AI social posts found that detectability "drops dramatically" once attackers use privately fine-tuned models ([arXiv 2506.09975](https://arxiv.org/abs/2506.09975)), so cover as many prompting and model styles as you can afford.
- **Make adversarial variants:** run paraphrase and humanizer passes over some AI comments.
- **Add noise to both classes:** inject typos, case changes, emoji and homoglyphs into human and AI examples alike, so that noise is never a class cue.

**Public corpora** can supplement your own data but should not replace it.

- **MultiSocial** (ACL 2025): 472,097 texts from Telegram, Twitter, Gab, Discord and WhatsApp, in 22 languages, from 7 generators. It is the closest public match to comments ([Zenodo](https://doi.org/10.5281/zenodo.13846152)).
- **RAID:** includes a Reddit domain and 11 adversarial attacks ([HF](https://huggingface.co/datasets/liamdugan/raid)).
- **AIGTBench:** covers Medium, Quora and Reddit with 12 LLMs ([arXiv 2412.18148](https://arxiv.org/abs/2412.18148)).
- **COLING 2025 Task 1 training set:** includes tweets ([GitHub](https://github.com/mbzuai-nlp/COLING-2025-Workshop-on-MGT-Detection-Task1)).
- **TweepFake:** 25,572 tweets, but from GPT-2-era bots ([arXiv 2008.00036](https://arxiv.org/abs/2008.00036)).

Their licenses were not checked during this research, so check each one before you use it. On size, Pangram's performance **saturated around 40,000 examples per domain** ([Pangram](https://arxiv.org/html/2402.14873v3)). A few tens of thousands of mirrored pairs per platform is therefore a realistic target for one person, not millions.

**Split by group, not at random.** Each AI comment and its human mirror go in the same split. So do all comments on one post, and all comments by one human author. Remove near-duplicates across every source with MinHash. Keep these held out for testing:

- at least one entire generator,
- at least one entire platform or community,
- the most recent time period.

## Fine-tune a small encoder, then hunt its false positives

The evidence favors supervised fine-tuning on in-domain short text, with a zero-shot score added as a backstop. On MultiSocial, the social-media benchmark closest to this project, the results were:

- **Fine-tuned detectors** reached **0.95–0.98 AUROC**. Even the small mDeBERTa-v3-base scored 0.954, despite median text lengths of only 9–18 words.
- **Zero-shot methods** (Binoculars and Fast-DetectGPT) scored about **0.72–0.74**.
- **Off-the-shelf detectors** were worse than chance. The old RoBERTa OpenAI detector scored 0.345 ([MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)).

The same paper adds an important caveat: when both the generator and the domain are unseen, Fast-DetectGPT and Binoculars can beat the fine-tuned models. The 2025 shared-task winners were mostly fine-tuned classifiers. The COLING English winner used DeBERTa-v3-base with auxiliary domain-classification heads. The runner-up used LoRA-tuned XLM-R and Qwen2.5-0.5B, and the top multilingual team ensembled Llama-3.2-1B and Gemma-2-2B with linguistic features ([COLING 2025 Task 1](https://arxiv.org/pdf/2501.11012v2)).

A practical recipe follows from this. These steps are synthesized from the sources above; no single study tested the combination.

1. **Baseline:** keep the TF-IDF plus logistic regression setup from the audit as a floor.
2. **Main model:** fine-tune `microsoft/deberta-v3-base` with a maximum of 128 tokens (comments are short), a learning rate around 2e-5, batch size 32, and 2–4 epochs with warmup.
3. **Step up if you have the GPU:** a QLoRA-tuned 0.5–2B decoder classifier (Qwen2.5, Llama-3.2 or Gemma-2). Use the MultiSocial settings: learning rate 2e-4 and an effective batch of 16 ([MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)).
4. **Ensemble:** combine the fine-tuned probability with a Binoculars or Fast-DetectGPT score, for robustness to unseen generators. Zero-shot performance depends heavily on which proxy model does the scoring ([arXiv 2601.15301](https://arxiv.org/html/2601.15301)), so fix that choice early.

**Hard-negative mining gives the biggest single gain.** Run the model over a large pool of comments known to be human. Add the ones it wrongly flags, along with their AI mirrors, to the training set, then retrain. Pangram's rounds of this cut its domain-weighted false-positive rate from **2.29% to 0.02%** ([Pangram](https://arxiv.org/html/2402.14873v3)). For a project that will be scored mostly on human comments, that is worth more than any change of architecture.

**Normalize Unicode before both training and inference.** That means NFKC normalization, homoglyph mapping and removal of zero-width characters. On RAID, a homoglyph attack dropped Originality from **85.0% to 9.3%** accuracy at a 5% false-positive rate, and synonym swaps cost Binoculars 36 points ([RAID](https://arxiv.org/html/2405.07940v2)). Normalization blocks the first attack for almost no cost, and synonym and paraphrase augmentation covers the second.

**Finally, calibrate.** Apply temperature scaling on a held-out calibration split, so that a score of 0.8 really means about 80%. The prevalence methods later in this report depend on calibrated probabilities.

## Judge the detector at 1% false positives on generators it never saw

At the COLING 2025 shared task, the official baseline scored **96.2% on the dev set and 74.9% on the blind test**, and only 14 of 36 teams beat that baseline ([COLING 2025 Task 1](https://arxiv.org/abs/2501.11012)). Any number from a random split of your own data will be optimistic in the same way.

**Metrics.** Report AUROC plus the true-positive rate at fixed false-positive rates of 1% and 5%. Set the thresholds on separate calibration data, as RAID and MultiSocial do ([RAID](https://arxiv.org/html/2405.07940v2); [MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)). The Chicago Booth study goes further and proposes "policy caps" such as flagging no more than 0.5% of human text ([Chicago Booth Review](https://www.chicagobooth.edu/review/2025/december/do-ai-detectors-work-well-enough-trust)).

The reason for focusing on low false-positive rates is simple arithmetic. Suppose true prevalence is 2% and the detector catches 90% of AI comments. At a 1% false-positive rate, about **65% of flags are correct**. At 5%, only **27%** are.

**Test sets.** Build these as separate suites:

- **Held-out generators.** MultiSocial saw lower performance on a Gemini model it held out of training.
- **Held-out platforms.** Training only on Discord cost Llama-3-8B **27% AUROC** on other platforms ([MultiSocial](https://aclanthology.org/2025.acl-long.36.pdf)).
- **Attacked text:** paraphrased, homoglyph and humanized variants.
- **Real human comments from before 2019.** The ICWSM 2026 audit found that state-of-the-art detectors produce "considerable false positives" on such posts. They also flagged liberal political discourse as AI more often than conservative discourse ([ICWSM 2026](https://ojs.aaai.org/index.php/ICWSM/article/view/42660)).
- **Non-native English writing.** Detectors have historically flagged it at high rates ([Liang et al.](https://arxiv.org/abs/2304.02819)).
- **Length buckets:** under 10, 10–25 and 25–50 words. Commercial tools lose accuracy below 50 words ([BFI WP 2025-116](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf)).

Report the false-positive rate for each subgroup, with bootstrap confidence intervals.

**Comparisons.** Score these against the same thresholds on your held-out comment set:

- Binoculars and Fast-DetectGPT as training-free baselines.
- The fine-tuned MultiSocial checkpoints as the strongest open baseline for social media.
- Pangram or GPTZero, if your budget allows. Independent 2025 tests found both had zero false alarms on 495 human passages ([Epoch AI](https://epoch.ai/data-insights/ai-text-detection-challenges)), but neither has been validated independently on comments under 50 words.

## Collect from Bluesky and YouTube through official channels

Access changed sharply in 2026, and that settles which platforms to use. The details below were current as of 11 Nov 2026.

| Platform | Access in Nov 2026 | Cost | Fit for a solo student |
|---|---|---|---|
| Bluesky | Jetstream, an official public stream that needs no authentication ([atproto.com](https://atproto.com/guides/streaming-data)) | Free | Primary source |
| YouTube | Data API v3. `commentThreads.list` costs 1 unit; the daily quota is 10,000 units ([Google](https://developers.google.com/youtube/v3/determine_quota_cost)) | Free within quota | Second source |
| Mastodon | 300 calls per 5 minutes per account ([Mastodon docs](https://docs.joinmastodon.org/api/rate-limits/)) | Free | Optional low-bot contrast |
| X | Pay-per-use: $0.005 per post read, $0.01 per user read, capped at 3M post reads a month ([X Docs](https://docs.x.com/x-api/getting-started/pricing)) | About $500 per 100k replies | Skip unless funded |
| Reddit | No new public API requests after 31 Oct 2026; full shutdown in March 2027 ([PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)) | n/a | Skip |
| Meta, TikTok | Research programs for vetted, institution-affiliated researchers; analysis in a cleanroom with limited export ([Meta](https://transparency.meta.com/en-gb/researchtools/meta-content-library/); [TikTok](https://developers.tiktok.com/doc/research-api-faq)) | Free | Only via a supervisor |

**The legal position favors official APIs over scraping.** US courts have ruled for scrapers twice:

- In *Meta v. Bright Data* (January 2024), the court held that Meta's terms did not prohibit scraping public data while logged off ([TechCrunch](https://techcrunch.com/2024/02/26/meta-drops-lawsuit-against-web-scraping-firm-bright-data-that-sold-millions-of-instagram-records/)).
- In *X Corp. v. Bright Data* (May 2024), X's terms-of-service claims were dismissed as preempted by copyright law ([Proskauer](https://newmedialaw.proskauer.com/2024/05/14/california-court-issues-another-noteworthy-decision-dismissing-breach-of-contract-and-tort-claims-in-web-scraping-dispute/)). That case is on appeal.

Commentators call these rulings "very fact specific" ([Zyte](https://www.zyte.com/blog/california-court-meta-ruling/)). They do not protect logged-in accounts or API keyholders, who are bound by the platform's terms.

Platform terms are tightening:

- Reddit's Rule 8 bans scraping without an authorized agreement.
- X's developer agreement forbids using its content to "fine-tune or train a foundation or frontier model" ([X Docs](https://docs.x.com/developer-terms/restricted-use-cases)). A small research classifier is arguably outside that clause, but don't rely on that without checking.
- Tools that scrape YouTube's web interface, such as `yt-dlp --write-comments`, sit outside YouTube's API terms. Avoid them.

If you are at an EU institution, the DSA Article 40 vetted-researcher route has accepted applications since 29 Oct 2025 ([ACM](https://www.acm.nl/en/publications/access-platform-data-vetted-researchers-under-dsa)). It is a long side track, not a starting point. One published rejection came after about 80 working days, because the applicant could not prove vetted-researcher status ([DSA Observatory](https://dsa-observatory.eu/2026/03/12/if-at-first-you-dont-succeed-reflections-on-a-rejected-art-40-dsa-data-access-request/)).

**Ethics and privacy.** The research notes did not verify jurisdiction-specific law, so treat these as conservative practice, not legal advice:

- Get an IRB or ethics determination before you collect anything.
- Treat handles and comment text as personal data.
- Pseudonymize author IDs at ingest with a keyed hash, and store the key separately.
- Honor deletions.
- Publish only aggregates. Never name an account as a "bot", because labelling real people as fake is the main way this project could cause harm.
- Share comment IDs, not text.

**Sampling.** Prevalence depends entirely on what you sample.

- Define the population explicitly, for example English-language YouTube videos from channels above a set subscriber count, in named categories, within a stated time window.
- Stratify by topic and community size, and draw posts at random within each stratum. Don't use only trending lists.
- Collect every comment on each sampled post, and re-crawl at 24 hours and 7 days.
- On Bluesky, take random time windows across several weeks. Jetstream's cursors are timestamps, so you can replay windows and resume collection ([Bluesky docs](https://bsky.network/docs/jetstream)).
- Pre-register your estimands.

State in the write-up that platforms remove spam before you collect, so **measured prevalence is a lower bound**. Store raw JSON in Parquet and a normalized table in DuckDB, keyed on platform plus comment ID. Keep near-duplicate hashes as flags rather than deleting those rows, because copy-paste spam is part of what you are measuring.

## Turn flags into a prevalence estimate with honest error bars

**First, decide what you are counting.** "AI-written" describes a text. "Bot" describes an account. Coordinated inauthentic behavior describes a group of accounts, and spam describes intent. These categories overlap without nesting inside one another:

- In the fox8 study, **76% of accounts posting "as an AI language model" were likely humans** pasting ChatGPT output ([Yang & Menczer](https://arxiv.org/pdf/2307.16336)).
- On Reddit, about **2% of users** adopted machine-generated text. Among those users, it made up roughly 10–40% of their comments ([La Cava et al.](https://arxiv.org/html/2510.07226v1)).
- Template spam bots post text that is not AI-generated, so an AI-text detector never sees them.

Report at least two numbers: **the share of comments that are AI-generated**, and **the share of comments posted by likely-automated or coordinated accounts**. Report the union only if you estimate it jointly.

**Account signals need a different toolkit, because the old one no longer works.** Botometer X uses archival, metadata-only scores from before mid-2023 ([OSoMe](https://osome.iu.edu/research/blog/introducing-botometer-x)). Detectors trained on TwiBot-22 degrade substantially on LLM-driven botnets such as BotSim-24 and GraphMind ([arXiv 2605.12512](https://arxiv.org/html/2605.12512)). For a comment sample, use behavioral features that work on any platform:

- account age,
- regular posting intervals, and activity with no daily sleep gap,
- near-duplicate text across accounts,
- accounts that comment on the same items within short time windows, following the coordination-network method of Pacheco et al. ([ICWSM 2021](https://ojs.aaai.org/index.php/ICWSM/article/view/18075)).

Keep comment-level, account-level and cluster-level scores as separate dimensions. Do not fuse them into one score.

**Never report the raw flag rate.** Simply counting flags ("classify and count") is biased when prevalence is low. Here is a worked example. Suppose true prevalence is 3%, the detector catches 80% of AI comments, and it flags 5% of human comments. The observed flag rate is then **7.25%, more than double the truth**, and two-thirds of the flags are wrong.

The fix is the Rogan-Gladen correction, also called adjusted classify-and-count (ACC): p̂ = (q − FPR)/(TPR − FPR), where q is the observed flag rate. With the numbers above it recovers 3.0% ([Forman 2008](https://doi.org/10.1007/s10618-008-0097-y)). The correction only works if the true-positive and false-positive rates are measured on your target population. Rates from essay benchmarks will not transfer.

**Build that validation set by hand-labelling a sample of the scraped comments.**

- Start with a simple random sample. About 400 comments gives a direct estimate within roughly ±2 percentage points at 5% prevalence.
- Add a score-stratified oversample of high-scoring comments, so the true-positive rate can be estimated precisely. Reweight by inverse inclusion probability (Horvitz-Thompson).
- Use 3–5 annotators who write with LLMs often. A majority vote of five such experts misclassified only **1 of 300 articles** ([Russell et al., ACL 2025](https://aclanthology.org/2025.acl-long.267)). That result was on articles, so expect much lower agreement on short comments.
- Give annotators a "can't tell" option and show them full account history.
- Plant known items to measure annotator accuracy.
- Report Krippendorff's α for each length bucket.

**Choose estimators and confidence intervals.** Use QuaPy, which implements the main methods and their confidence intervals ([QuaPy](https://github.com/HLT-ISTI/QuaPy)). Report four estimates side by side:

- adjusted classify-and-count,
- probabilistic classify-and-count on calibrated scores,
- SLD (EM-based prior adjustment), only with a calibrated classifier, since it degrades otherwise ([Esuli et al.](https://iris.cnr.it/retrieve/7ff467d4-8025-49c0-8ca8-2821579fdf71/Esuli_Molinari_Sebastiani_ACM%20TOIS-2022-postprint.pdf)),
- the direct estimate from the labelled sample.

If these disagree widely, the error rates are not transferring or the scores are miscalibrated.

Uncertainty in the false-positive rate dominates the confidence interval. Suppose that rate is estimated from 200 labelled human comments. The standard error is about 1.5 points, which propagates to **roughly ±4 points on a 3% estimate**. To narrow that, either run at a very low false-positive rate or label more human comments.

Use a two-level bootstrap: resample both the validation labels and the target comments, clustering by account or thread. A Bayesian model with Beta priors also works. Under low prevalence, the Rogan-Gladen estimate tends to overshoot, and naive bootstrap intervals cover the true value erratically ([arXiv 2111.02910](https://arxiv.org/pdf/2111.02910)).

**Check that the corrected estimate is stable across thresholds.** A correctly adjusted estimate should barely move when the threshold changes. In La Cava et al.'s uncorrected study, by contrast, moving the threshold from 0.99 to 0.95 roughly doubled the result.

**Report comments too short to score as their own stratum.** Dropping them changes the denominator.

**Compare your result with published estimates:**

| Study | What it measured | Estimate |
|---|---|---|
| La Cava et al., Reddit 2022–2024 ([arXiv 2510.07226](https://arxiv.org/html/2510.07226v1)) | AI-generated comments | "Marginally present"; monthly peaks of about 8.5% |
| Pangram vendor study, 2026 ([Pangram](https://www.pangram.com/blog/ai-in-your-feed)) | Posts across five platforms | 13.8% fully AI; Reddit 4.4% |
| Originality.ai, 2025 ([Originality.ai](https://originality.ai/pt/blog/ai-reddit-posts-study)) | Reddit posts | About 15% |

Posts run higher than comments. If you find more than 10% across all comments without targeting a brigaded thread or a spam channel, audit your false positives before you publish anything.

## The build order for one developer

The phases below put the work in dependency order. Each gate is a pass/fail check that stops a phase from building on a broken one.

| # | Phase | Deliverable | Gate before moving on |
|---|---|---|---|
| 1 | Retire the uploads | archive_3 deleted from the pipeline; archive_5 kept only as a smoke test; archive_4 normalized, metadata dropped | Backtick/whitespace artifacts gone |
| 2 | Ethics and access | IRB determination; YouTube API key; Bluesky Jetstream consumer | Written data-management plan |
| 3 | Human corpus | archive_4 + pre-2019 YouTube comments + recent target-platform comments, deduplicated | Grouped splits by thread/author |
| 4 | AI corpus | Mirrored comments from 8–10 generators with varied prompts, decoding, personas and humanizer passes | One generator and one platform held out |
| 5 | Baselines | TF-IDF+LR, Binoculars/Fast-DetectGPT, a MultiSocial checkpoint | Numbers at 1% and 5% FPR |
| 6 | Main model | DeBERTa-v3-base, 128 tokens, Unicode normalization, calibration | Beats baselines on held-out generator |
| 7 | Hard negatives | 2–3 mining rounds over a large human pool | Target-platform FPR ≤1% |
| 8 | Robustness audit | Subgroup FPR by length, platform, ESL, politics; attack suite | No subgroup far above global FPR |
| 9 | Target collection | Stratified, pre-registered Bluesky + YouTube sample with author enrichment | Sampling frame and seeds logged |
| 10 | Validation labels | About 400 random + score-stratified oversample, 3–5 annotators | α reported per length bucket |
| 11 | Quantification | ACC/PCC/SLD + direct estimate; two-level bootstrap CIs; account-level signals separately | Estimates agree and are stable across thresholds |
| 12 | Write-up | Two headline numbers with CIs, per platform and length stratum, stated as lower bounds | No individual accounts named |

Phases 2 and 9 can run in parallel with phases 3–8, because collection takes calendar time while training takes compute time.

## Conclusion

The likeliest failure here is a confident wrong number, not a weak model. A detector with a 5% false-positive rate, run over comment sections where perhaps 2–3% of text is AI, produces a headline dominated by misclassified humans. A pipeline built on archive_3 or archive_5 would publish a figure of 17–33% with nothing behind it. That reframes what "effectively train" means. The goal is not a high benchmark score. It is a false-positive rate that is low, measured on your own target population, and small enough to correct for. That is why hard-negative mining and the hand-labelled validation sample matter more than which architecture you choose.

Two caveats shape how the result should be presented. First, any estimate is a lower bound for a length-qualified population on one or two platforms at one point in time. Platforms delete spam first, short comments resist detection, and privately fine-tuned generators evade classifiers. Second, the detector will age as new models ship, so the validation step should be repeated for every measurement period rather than done once. A modest, well-bounded number with honest error bars, phrased in the form "X% (95% CI a–b) of scorable comments on platform P in window W", is more defensible than any headline. On a topic this prone to alarmist claims, that defensibility is what makes the project worth doing.

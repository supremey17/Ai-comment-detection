# Audit of the three uploaded datasets for an AI-vs-human social-media-comment detector

Measured on 2026-11-11 with pandas 3.0.5 and scikit-learn 1.9.1. The data was extracted to the scratchpad and was not copied into the repo.

Naming used below:
- **A3** = `archive_3.zip` -> `ai_human_content_detection_dataset.csv`
- **A4** = `archive_4.zip` -> `train.xlsx`
- **A5** = `archive_5.zip` -> `ai_vs_human_text_2026.csv`

"[Measured]" means I computed the number from the file with the analysis scripts in `/tmp/claude-0/-home-user-Ai-comment-detection/487b0154-22ad-5df6-93c4-8475ac181bd5/scratchpad/scripts/`:
- `overview.py`: schema
- `deep.py`: lengths, style, duplicates, templates
- `baseline.py`: TF-IDF/LR cross-validation and cross-dataset tests
- `extra.py`: metadata checks

The baseline model is one setup throughout:
- **Features:** word 1-2-gram TF-IDF plus char_wb 2-5-gram TF-IDF (200k features max, sublinear TF).
- **Classifier:** LogisticRegression with C=4.
- **Validation:** stratified 5-fold CV unless stated otherwise.

## Q1. File inventory: rows, columns, labels, class balance, length, language

### Takeaway
Only A3 and A5 have an AI/human label.
- **A3:** 1,367 rows, balanced 684/683.
- **A5:** 2,000 rows, 1,334 human vs 666 AI, but only 966 unique texts.
- **A4:** 27,481 English tweets with a 3-way *sentiment* label and **no AI/human label at all**. It is a human-only tweet corpus (the Kaggle Tweet Sentiment Extraction data).

### Cited Findings

**A3 (`ai_human_content_detection_dataset.csv`, 1,409,940 bytes, zip timestamp 2025-07-28)**
- **Shape:** 1,367 rows × 17 columns. [Measured]
- **Columns:**
  - `text_content` (str), `content_type` (str), `label` (int 0/1).
  - 14 precomputed numeric features: `word_count`, `character_count`, `sentence_count`, `lexical_diversity`, `avg_sentence_length`, `avg_word_length`, `punctuation_ratio`, `flesch_reading_ease`, `gunning_fog_index`, `grammar_errors`, `passive_voice_ratio`, `predictability_score`, `burstiness`, `sentiment_score`. [Measured]
- **Label:** 0 = 684, 1 = 683. The file has no data dictionary, so it does not say which value means AI. The Kaggle source most likely uses 1 = AI, but I could not verify that (see Q6). [Measured]
- **Missing values:** `text_content` has none. The features have some: `flesch_reading_ease` 79, `sentiment_score` 54, `gunning_fog_index` 35, `passive_voice_ratio` 31. [Measured]
- **`content_type`** has 8 values: academic_paper 189, news_article 185, blog_post 183, creative_writing 172, article 171, essay 161, social_media 154, product_review 152. [Measured]
- **Length:**
  - Label 0: chars min 27 / median 881 / p90 1,786 / max 2,889; words 3 / 131 / 267 / 438. [Measured]
  - Label 1: chars 14 / 882 / 1,848 / 2,966; words 3 / 130 / 271 / 443. [Measured]
  - The `social_media` subset has median 17.5 words, max 39. [Measured]
- **Language:** English-looking word salad (see Q3).

**A4 (`train.xlsx`, 2,188,421 bytes, zip timestamp 2024-10-03, one sheet "Sheet1")**
- **Shape:** 27,481 rows × 6 columns: `textID`, `text`, `sentiment`, `Time of Tweet`, `Age of User`, `Country`. [Measured]
- **Sentiment:** neutral 11,118, positive 8,582, negative 7,781. There is **no AI/human label**. [Measured]
- **Missing values:** 1 missing `text` (textID `fdb77c3752`). [Measured]
- **Length:** chars min 3 / median 64 / p90 122 / max 159; words 1 / 12 / 23 / 33. [Measured]
- **Language:** English.

**A5 (`ai_vs_human_text_2026.csv`, 696,551 bytes, zip timestamp 2026-05-03)**
- **Shape:** 2,000 rows × 9 columns: `text_id`, `label` (human/ai), `source_model`, `domain`, `text_content`, `topic_hint`, `word_count`, `avg_sentence_length`, `generation_method`. There are no missing values. [Measured]
- **Label:** human 1,334 (66.7%), ai 666. [Measured]
- **`source_model`:** human 1,334, gemini-2.0 333, gpt-4o 333. [Measured]
- **`generation_method`:** "template+human_variation" covers all 1,334 human rows; "style_simulation" covers all 666 AI rows. This column is a perfect label proxy. [Measured]
- **`domain`:** social 674, academic 669, news 657. Each model contributes exactly 111 rows per domain. [Measured]
- **`topic_hint`:** 30 topics (for example "gene editing ethics", "student debt crisis"), 52-82 rows each. [Measured]
- **Length:**
  - AI: chars 152 / 295 / 359 / 385; words 23 / 39 / 47 / 50. [Measured]
  - Human: chars 123 / 253 / 369 / 398; words 19 / 36 / 49 / 52. [Measured]
- **Language:** English.

### Inferences
- A4 cannot train a binary AI detector on its own. It is useful only as a source of real human negatives, or as an out-of-domain false-positive test set.
- A5's AI/human imbalance (1:2) and its leaky metadata columns (`source_model`, `generation_method`) must be dropped or ignored before training.

### Gaps
- A3's label semantics (which value is AI) are undocumented in the file, and I could not fetch the Kaggle page to confirm them.

## Q2. Representative samples per class (truncated)

### Takeaway
- **A3:** both classes are indistinguishable random word salad.
- **A5:** both classes are short template sentences with the topic slotted in. The "human" academic and news texts read exactly like LLM prose.
- **A4:** real 2009-era tweets.

### Cited Findings

**A3 label 0** [Measured]:
- "Open no cold. Field challenge candidate few executive relationship. Always nor artist network various road knowledge.\nI office agree people get respond environmental. List my moment..."
- "She attorney response total. Morning even class money.\nMust region thus participant outside price..."

**A3 label 1** [Measured]:
- "Last while seven trouble set. Owner dark usually blue.\nTravel he doctor catch. Weight girl citizen within..."
- "Fly speech car hope picture make wear. Value near ground run remember. Hospital street send improve drug full even."

**A3, content_type=social_media** [Measured]:
- Label 0: "Third few low hard peace paper pass front. Mean mention possible stuff. Unit get relationship."
- Label 1: "Analysis material voice between everyone dream.\nYear six run. Role produce yeah through. Culture nice bar rock stay."

**A4** (all human) [Measured]:
- "I`d have responded, if I were going" (neutral)
- "Sooo SAD I will miss you here in San Diego!!!" (negative)
- "really? ur a brave lady walkin around those lions then.  any other book suggestions?" (positive)
- "hahaha u can use my phone everywhere... just use a wi-fi" (positive)
- "Today is a sad day for Tucson: http://tr.im/mQ39 I shall miss the sour cream apple pie..." (negative)

**A5 social, ai** [Measured]:
- "Really fascinating to see how quickly things are moving on economic inequality trends. There's so much potential here if we approach it thoughtfully. What are your thoughts?"
- "Spent the morning reading about data privacy legislation and came away genuinely hopeful. Yes, the challenges are real, but so are the opportunities. Let's focus on solutions."

**A5 social, human** [Measured]:
- "so cybersecurity threats to infrastructure happened and my timeline is UNHINGED. some of you need to calm down and some of you need to be WAY more concerned tbh"
- "just saw someone arguing that urban housing shortages is fine and i'm genuinely baffled. do people not understand how this works?? 😭"
- "idk why cryptocurrency market volatility keeps coming up but every time it does the discourse gets worse. we need better conversations about this fr"

**A5 academic** [Measured]:
- Human: "The theoretical underpinnings of cryptocurrency market volatility have evolved considerably over the past two decades. What began as a niche area of inquiry has grown into a robust field..."
- AI: "The concept of supply chain disruptions occupies a contested space within contemporary scholarly discourse..."

**A5 news, human** [Measured]: "Officials confirmed on Thursday that aging population challenges. The announcement came after weeks of speculation..." This is ungrammatical, because the template slot expects a clause but got a noun phrase.

### Inferences
- A5's "human" texts are not real human writing. They are script-filled templates, almost certainly authored by an LLM or a programmer. The "human" academic and news templates are stylistically indistinguishable from the "AI" ones.

### Gaps
- None.

## Q3. Data quality: duplicates, missing values, synthetic/templated signals, leakage

### Takeaway
- **A3 is pure random-word noise.** Its labels carry no learnable signal: TF-IDF CV accuracy is 0.500, exactly chance.
- **A5 is fully templated.** It has 18 AI templates and 29 human templates × 30 topics, and 1,034 of 2,000 rows are exact duplicates. Any model learns template identity, not "AI-ness".
- **A4 is real data**, but its demographic columns are fabricated.
- **No text is shared between any two files.**

### Cited Findings

**A3: synthetic word salad**
- **Vocabulary:** the whole file uses a vocabulary of only **971 distinct lowercase word types** over 191,641 tokens. The most frequent words are each almost equally common (tv 245, from 236, there 235, level 234, hair 233, ...). That flat distribution is consistent with the Python Faker `sentence()`/`paragraph()` lorem generator, not natural text, which follows Zipf's law. [Measured]
- **No natural punctuation:** both classes have 0% rows with `?`, `!`, apostrophes, digits, emoji, URLs, @mentions, hashtags or non-ASCII characters. 92-94% of rows contain newlines. [Measured]
- **Duplicates:** 0 exact and 0 normalized duplicates. [Measured]
- **Class means of the precomputed features are nearly identical.** Label 0 vs 1: lexical_diversity 0.969/0.967, avg_sentence_length 5.50/5.47, flesch 52.18/52.18, grammar_errors 1.56/1.52, burstiness 0.442/0.412, predictability 63.8/61.7. [Measured]
- **Predictability test** (separability is near chance on every feature):
  - TF-IDF+LR: acc **0.500 ± 0.013**, AUC 0.512.
  - HistGradientBoosting on the 14 precomputed features: acc 0.520 ± 0.020, AUC 0.521.
  - Same plus content_type: acc 0.511.
  - Handcrafted style features: acc 0.505, AUC 0.499.
  - Best single feature: burstiness, AUC 0.543.

  [Measured]
- **Description match:** the description of this kind of Kaggle file ("synthetic dataset of text samples labeled for AI vs human content detection", with a content_type field) matches the Kaggle dataset "AI vs Human Content Detection 1000+ record in 2025" by pratyushpuri. — [Kaggle search result](https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025) (the page itself could not be fetched; it is JS-rendered and the API returned 404 without auth)
- **A published paper used a 1,367-sample, 17-feature dataset** with lexical diversity, Flesch, Gunning Fog, burstiness and predictability. That matches A3's shape and suggests the file has been used in at least one 2025 paper. — [Atlantis Press PDF (search snippet)](https://www.atlantis-press.com/article/126024937.pdf). I could not extract the PDF text, so the link to this file is unconfirmed.

**A5: templated and heavily duplicated**
- **Duplicates:** only **966 unique texts / 2,000** (1,034 exact-duplicate rows; 1,134 after lowercase and punctuation normalization). [Measured]
- **Unique texts per class:** ai 387, human 579. No text appears under both labels. [Measured]
- **Most repeated text:** one "human" academic text appears **11 times**: "Recent scholarship on gig economy labor rights has largely focused on surface-level manifestations...". [Measured]
- **Templates:** after replacing `topic_hint` with `<T>`, there are only **18 distinct AI templates** and **29 distinct human templates**. [Measured]
  - Largest human templates: "The theoretical underpinnings of <T>..." ×117, "Recent scholarship on <T>..." ×111, "Local authorities are responding to reports that <T>..." ×91.
  - Largest AI templates: "The concept of <T> occupies a contested space..." ×46, "I've been thinking a lot about <T> lately..." ×45.
- **Each AI template belongs to exactly one model.** There are 9 templates for gpt-4o and 9 for gemini-2.0, so `source_model` is a deterministic function of the template. Example: "a growing body of research" is gemini-2.0 ×42 and gpt-4o ×0. [Measured]
- **The "human_variation" is shallow.**
  - Social human texts are 100% lowercase-initial, versus 3-4% for academic and news human texts.
  - Emoji appear in 9.2% of human rows and 0% of AI rows. Only three emoji are used: 🧵 67, 👇 67, 😭 56. They are fixed parts of 2 templates.
  - ALL-CAPS words appear in 9.7% of human rows and 0% of AI rows.
  - The only non-ASCII character in AI rows is the em dash "—" (37 occurrences).

  [Measured]
- **Predictability test:**
  - TF-IDF+LR, random 5-fold: acc **1.000**, AUC 1.000.
  - Grouped by exact text: 1.000.
  - Grouped by topic: 1.000.
  - Handcrafted style features (length, caps, lowercase start, non-ASCII, punctuation counts): acc 0.993, AUC 1.000.
  - Only `word_count` + `avg_sentence_length` (the provided columns): acc 0.847, AUC 0.926.

  [Measured]
- **Generalization to unseen templates:** with GroupKFold over the 38 template-prefix groups, TF-IDF accuracy falls to **0.359 ± 0.137 (AUC 0.345)**, which is worse than chance and worse than the 0.667 majority baseline. Per domain: social 0.460, academic 0.084, news 0.267. The model memorizes templates and does not learn a transferable AI signal. [Measured]
- **Matching Kaggle dataset:** "AI vs Human Text Classification Dataset 2026" (alitaqishah) describes 2,000 samples across academic writing, news articles and social media posts, generated via "style simulation prompts designed to mimic domain-specific writing conventions", with GPT-4o and Gemini 2.0 named. This matches A5's row count, domains, models and `generation_method` values. — [Kaggle (search-engine summary)](https://www.kaggle.com/datasets/alitaqishah/ai-vs-human-text-classification-dataset-2026). The page body could not be fetched.

**A4: real tweets with fabricated metadata**
- **Duplicates:** 0 exact duplicates and 247 near-duplicates after normalization. 1 missing text. [Measured]
- **Fabricated demographics:**
  - `Time of Tweet` cycles exactly morning/noon/night by row index.
  - `Age of User` cycles exactly through 0-20, 21-30, 31-45, 46-60, 60-70, 70-100 by row index.
  - `Country` is near-uniform over 195 countries (127-149 rows each).

  [Measured]
- **Text artifacts:**
  - 8,807 rows use a backtick "`" as the apostrophe, versus 596 with a straight "'".
  - 10,975 rows start with a leading space.
  - Usernames appear removed (for example "As of today, _a_Bo0 and I ...").
  - 0 emoji.

  [Measured]
- **Origin:** the row count, columns (`textID`, `text`, `sentiment`) and sentiment split match the Kaggle "Tweet Sentiment Extraction" competition training set. That set has 27,481 training rows and 3,534 test rows, with columns textID, text, selected_text and sentiment, and its phrases come from Figure Eight's Data for Everyone platform. It also has one known missing row and a split of about 40% neutral, 31% positive and 28% negative. — [cengc13 Kaggle blog](https://cengc13.github.io/website/kaggle/2020/06/01/kaggle-tweet-start-blog.html); [GitHub kelvng/Tweet-Sentiment-Content-Extraction](https://github.com/kelvng/Tweet-Sentiment-Content-Extraction); [arXiv 2104.09777](https://arxiv.org/pdf/2104.09777)
- **Repackaged derivative:** the same `Time of Tweet`/`Age of User`/`Country` columns appear in a repackaged "Sentiment Analysis Dataset" (a Gigasheet sample also lists Population, Land Area and Density columns). — [Gigasheet sample](https://www.gigasheet.com/sample-data/sentiment-analysis-dataset). The Kaggle uploader is most likely abhi8923shriv/sentiment-analysis-dataset, but I could not fetch its page to confirm.

**Cross-file overlap**
- Exact-match overlap after lowercase and strip: A3∩A4 = 0, A3∩A5 = 0, A4∩A5 = 0. [Measured]

### Inferences
- **A3 should be discarded.** Its texts are random words and its labels are random with respect to the text. Any reported accuracy on it is noise, and its "features" are just statistics of that noise.
- **A5's 100% in-distribution accuracy is an artifact.** The label is recoverable from the template, from lowercase/emoji/caps quirks, and from `generation_method`. Training on A5 would teach a social-media detector that lowercase plus slang means human and polished means AI, a shortcut that real AI comments (often prompted to be casual) will defeat.
- **A4's metadata columns must never be used as features.** They are deterministic row-index artifacts.

### Gaps
- I could not verify exactly how A3 was generated (Faker or something else) because the Kaggle page was unreachable. The Faker hypothesis rests on the measured vocabulary and frequency pattern.

## Q4. Cross-dataset generalization

### Takeaway
Models trained on A3 or A5 do not transfer.
- A3 to A5 and A5 to A3 are near chance (AUC 0.578 and 0.529).
- On 27,480 real human tweets, an A3 model flags 32.5% as AI and an A5 model flags 17.5%.

### Cited Findings
- **Train A3, test A5:** acc 0.632, AUC 0.578, predicted-AI rate 0.251. The accuracy is inflated by A5's 2:1 human majority. [Measured]
- **Train A5, test A3:** acc 0.516, AUC 0.529. [Measured]
- **False-positive rate on A4** (all human): [Measured]
  - Train A3: 32.5% predicted AI (mean p = 0.429).
  - Train A5: 17.5% (mean p = 0.342).
  - Train on A5 social domain only: 16.7% (mean p = 0.340).

### Inferences
- A 17-33% false-positive rate on genuine human tweets would make a comment-moderation detector unusable. Neither labelled set gives a usable starting model.

### Gaps
- The A4 tweets are from about 2009 (Figure Eight data). How well a model does on modern human comments (2024-2026 Reddit/YouTube/X) was not measured.

## Q5. Resemblance to real social media comments

### Takeaway
- **A4 is the only real social-media text.** It is short (median 12 words) and informal, with typos, "u", "!!!", "..." and URLs. It is 2009-era and has no emoji.
- **A5 imitates social media in a cartoonish, templated way**, about 674 rows from 9 social templates per class.
- **A3 does not resemble any real text.**

### Cited Findings
- **A4 informal markers, share of rows:** [Measured]
  - `!` 29.6%, `?` 10.8%, `..` 18.7%
  - ALL-CAPS word 11.1%, lowercase start 21.0%, digits 17.1%
  - URL 4.7%, hashtag 1.9%, @mention 0.3% (usernames were stripped)
  - emoji 0%
- **A5 social domain:**
  - 100% lowercase-initial for human and 0% for AI.
  - Lengths 19-52 words, longer than typical comments.
  - Human social texts use canned slang ("tbh", "fr", "idk", "rn", "lol", "??", 🧵👇 😭). AI social texts are polished, upbeat and end with prompts like "What are your thoughts?"

  [Measured]
- **A3 `social_media` rows** are 3-39 words of word salad with no punctuation beyond periods. [Measured]

### Inferences
- For short informal comments, only A4 (as human negatives) is in-domain. It needs normalizing to match modern text: convert "`" to "'", strip leading spaces, and decide how to handle URLs. Otherwise a model learns "backtick = human".

### Gaps
- None of the three files contains modern emoji-rich comments, Reddit/YouTube/TikTok style, non-English text, or code-switching.

## Q6. Generator LLMs represented

### Takeaway
- **A5** names gpt-4o and gemini-2.0 (333 rows each). In practice these are 9 fixed templates per "model", so the outputs carry essentially no real model signal.
- **A3** names no models.
- **A4** has no AI text.

None of the files covers 2025-2026 models: GPT-4.1/GPT-5 class, Claude 3.5-4.x, Gemini 2.5+, Llama 3/4, Qwen, DeepSeek or Grok.

### Cited Findings
- A5 `source_model` takes only the values human, gemini-2.0 and gpt-4o, and each model maps to 9 disjoint templates. [Measured]
- The Kaggle description of the matching 2026 dataset names GPT-4o and Gemini 2.0 and "style simulation prompts". — [Kaggle (search summary)](https://www.kaggle.com/datasets/alitaqishah/ai-vs-human-text-classification-dataset-2026)
- A3's Kaggle counterpart is described as synthetic. No generator model is documented in what I could retrieve. — [Kaggle](https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025)
- A4's text comes from Figure Eight "Data for Everyone" via the 2020 Kaggle Tweet Sentiment Extraction competition, which is all human text. — [cengc13 blog](https://cengc13.github.io/website/kaggle/2020/06/01/kaggle-tweet-start-blog.html)

### Inferences
- Even if the A5 "gpt-4o"/"gemini-2.0" rows were genuinely produced by those models, the template collapse (387 unique AI texts) means far too few independent generations to learn model style.

### Gaps
- I could not confirm the Kaggle descriptions verbatim because the pages are JS-rendered and the Kaggle API returned 404 without credentials.

## Q7. Recommendations: what to use, how to merge, what is missing

### Takeaway
- **Do not use A3** for anything.
- **Do not use A5 for training.** At most, keep its 579 unique human and 387 unique AI texts as a small sanity-check set, and drop `source_model` and `generation_method`.
- **Use A4** (de-duplicated, normalized, metadata dropped) only as a source of real short human negatives.

A real detector needs new AI-positive data: modern LLMs prompted to write *short, casual comments*, including paraphrased and humanized variants, paired with modern human comments from the same platforms and topics.

### Cited Findings
- A3 measured signal is zero: CV acc 0.500. [Measured]
- A5's in-distribution accuracy is 1.000, but 0.359 on unseen templates. [Measured]
- A4 provides about 27.2k unique real human short texts (27,480 non-null minus 247 near-duplicates). [Measured]
- The Defactify / "Comprehensive Dataset for Human vs. AI Generated Text Detection" pairs 58k+ NYT articles with outputs from Gemma-2-9B, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large and GPT-4o. Its baseline human-vs-AI accuracy is about 58%. It is long-form news, not comments. — [arXiv 2510.22874](https://arxiv.org/pdf/2510.22874)
- LLMTrace is a bilingual (EN/RU) classification corpus with a per-row generator (for example gemini-2.0-flash) and a grouping ID linking human sources to their AI rewrites. — [Hugging Face LLMTrace_classification](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification/blob/main/README.md)
- PAN 2026 runs a Voight-Kampff human-vs-machine task with about 20k labelled samples. Participant augmentation used gpt-3.5-turbo, gemini-2.0-flash, llama-3.3-70b-instruct and others. — [CEUR-WS paper](https://ceur-ws.org/Vol-4283/paper397.pdf)

### Inferences

**Merge and normalization plan** (if any of the three files is kept):
- **Unify labels:** use `is_ai` ∈ {0,1}. A5 maps "ai" to 1. A4 is all 0.
- **Drop leaky metadata:** all metadata columns, namely A5's `source_model`, `generation_method`, `word_count` and `avg_sentence_length`, and A4's Time/Age/Country.
- **Normalize text:**
  - Convert "`" to "'".
  - Strip and collapse whitespace.
  - Unify em dashes and quotes, or deliberately keep them as features, but consistently across sources.
  - Optionally mask URLs and @handles.
- **Deduplicate:** exact and near-duplicates (MinHash or normalized-string) across all sources.
- **Split by group:** split by source, template or topic, not at random, and always report a held-out-source score.

**What is missing:**
1. Real AI-written short comments (5-60 words) on social platforms.
2. Recent generators (GPT-4o-mini/4.1/5, Claude 3.5-4.x, Gemini 2.x-3, Llama 3.x/4, Qwen 2.5/3, DeepSeek V3/R1, Mistral, Grok).
3. Varied prompts and personas, including "write like a redditor", typos and emoji.
4. Adversarial variants: paraphrased, humanizer tools, human-edited AI, and AI-edited human text.
5. Modern human comments (2022-2026) from Reddit, YouTube and X, both pre-LLM-era and recent, matched on topic.
6. Non-English and code-switched text, if needed.
7. Metadata for group splits (platform, thread, author).

**Candidate public sources to evaluate** (other researchers may cover these; I did not verify them in this session beyond the links above):
- RAID, M4/M4GT-Bench, MAGE/DeepfakeTextDetect, HC3, Ghostbuster, the Kaggle "LLM - Detect AI Generated Text" (DAIGT) corpora, and TweepFake (real GPT-2-era bot tweets).
- PAN 2025/2026 Voight-Kampff data, LLMTrace, and Defactify (cited above).
- Pushshift-era Reddit dumps or YouTube comment datasets as human short-text sources.

The most reliable fix is to generate one's own AI comments: prompt several current LLMs with the *same* real threads that the human comments reply to, which controls for topic.

### Gaps
- I did not verify the licenses, sizes or comment-likeness of the candidate external datasets; my budget for this session went to the hands-on file audit.
- The Kaggle pages for all three uploads could not be fetched directly: JS-rendered, and the API and Croissant endpoints returned 404 without authentication. The origin attributions above rely on search-engine summaries plus matching schemas and counts.

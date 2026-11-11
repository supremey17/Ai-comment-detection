# Data Collection and Legal Constraints for Measuring AI/Bot Comment Prevalence (as of 11 Nov 2026)

Scope: how a solo developer or student can legally and practically collect social media comments, platform by platform, and what the legal, ToS and ethical constraints are. Detection modeling is out of scope. Research date: 2026-11-11. Each finding gives its source. Where a claim rests on background knowledge that I did not re-verify in this session, it is marked **[unverified this session]** and kept out of "Cited Findings".

---

## Q1. Platform-by-platform data access: eligibility, cost, rate limits, available metadata

### Takeaway
As of November 2026 the access picture has changed a lot. **Reddit is closing its public Data API** (no new requests after 31 Oct 2026, full shutdown March 2027). **X is pay-per-use** at about $0.005 per post read and $0.01 per user read, capped at 3M post reads a month. **YouTube Data API v3 stays free** within 10,000 units a day, and a comment-thread page costs 1 unit. **Bluesky (Jetstream) and Mastodon are free and open**. **Meta Content Library and TikTok Research API are free** but limited to vetted, institution-affiliated researchers with restricted export. For a low-budget solo project, Bluesky and YouTube are the most practical starting points.

### Cited Findings

**Reddit**
- Reddit announced on 30 Sep 2026, in r/modnews and r/redditdev, that the public Data API will shut down in phases. New requests for public API access stop on **31 Oct 2026**. RSS feeds end on **13 Nov 2026**. Access starts closing on **12 Jan 2027** for apps and users that have not registered or responded to outreach. The remaining public API access closes in **March 2027** (no exact day given) — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- Unauthenticated `.json` endpoints were already shut down, per a June 2026 moderator post — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- The replacement is Devvit (Reddit's TypeScript developer platform), which has no Python support and none on the roadmap. Reddit offers a "PRAW migration guide" and a $1,000-per-app migration bounty (registration until 30 Nov 2026). Exemptions are limited to partners with written agreements — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- Reddit's Rule 8 prohibits scraping "without an authorized agreement". Reddit's licensing deals include a Compliance API that notifies licensees of deletions — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- Third-party guides say academic research goes **only** through the "Reddit for Researchers" program, and that the help center names it as the only authorized route for research. The program was a beta built with OpenMined for non-commercial, privacy-preserving access — [apiserpent](https://apiserpent.com/blog/free-reddit-api); [OpenMined](https://openmined.org/blog/announcing-our-partnership-with-reddit-to-expand-privacy-preserving-researcher-access/); [OpenMined: beta growing](https://openmined.org/blog/reddit-for-researchers-beta-program-is-growing/)
- An unverified user comment from February 2026 on r/redditdev says "only PIs are eligible to apply" to Reddit for Researchers. A late-2025 thread reported that the admin who ran the program had left Reddit — via [search summary of r/redditdev mirrors](https://r.datuan.dev/r/redditdev/comments/1oug31u/introducing_the_responsible_builder_policy_new/nobwwms/?context=3) (weak evidence)
- **Pushshift**: public access was revoked in May 2023 and is now restricted to verified moderators — [dev.to guide](https://dev.to/agenthustler/reddit-data-api-2026-after-the-pricing-change-heres-what-developers-actually-use-4o2h); [xpoz](https://www.xpoz.ai/blog/comparisons/best-reddit-api-alternatives-2026/)
- **Arctic Shift** is cited by third-party guides as a free archive of Reddit data from 2005 to present for academic research and bulk archives, with a 1-2 month lag. I could not verify its legal standing under Reddit's terms — [xpoz](https://www.xpoz.ai/blog/comparisons/best-reddit-api-alternatives-2026/)

**YouTube Data API v3**
- `commentThreads.list` costs **1 quota unit** per call. The default project quota is **10,000 units a day**, and invalid requests also cost at least 1 unit — [Google: Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost)
- Each page of results is a separate request. The quota resets at midnight Pacific Time, and going over it returns a 403 — [Phyllo](https://www.getphyllo.com/post/is-the-youtube-api-free-in-2026-quota-limits-costs-when-to-pay)
- Third-party guides agree that `comments.list` (for replies) also costs 1 unit — [zernio](https://zernio.com/blog/youtube-comments-api)

**X / Twitter**
- On **6 Feb 2026**, pay-per-use became the default pricing and the free tier closed to new signups. Basic subscribers were auto-migrated to pay-per-use starting **1 Jun 2026**. Enterprise remains for higher volume — [Open Terms Archive](https://opentermsarchive.org/en/memos/x-replaces-its-dollar200-entry-level-api-plan-with-a-pay-per-use-model/); [postproxy](https://postproxy.dev/blog/x-api-pricing-2026)
- Official price list (no subscription; credits are bought upfront):

  | Resource | Price |
  |---|---|
  | Post read | $0.005 per post |
  | User read | $0.010 per user |
  | Followers/Following read | $0.010 |
  | Counts (recent) | $0.005 |
  | Counts (all) | $0.010 |
  | Owned reads | $0.001 |

  Cap: **3 million post reads per monthly billing cycle**; above that you need Enterprise. Charges are deduplicated within a 24-hour UTC day (a "soft guarantee"). New accounts get a $20 credit after saving a card and making a first purchase, plus a match of the first auto-recharge up to $50 — [X Docs: Pricing](https://docs.x.com/x-api/getting-started/pricing) (fetched 2026-11-11; the page shows no "last updated" date)
- The official pricing page does not mention academic or research access — [X Docs: Pricing](https://docs.x.com/x-api/getting-started/pricing)
- Third-party sources disagree on some figures: one gives a 2M-read cap, others say 3M. The official page says 3M — [Open Terms Archive](https://opentermsarchive.org/en/memos/x-replaces-its-dollar200-entry-level-api-plan-with-a-pay-per-use-model/) vs [X Docs](https://docs.x.com/x-api/getting-started/pricing)
- X has a help page on "DSA Data Access for Vetted Researchers" (the EU Article 40 route) — [X Help](https://help.x.com/en/rules-and-policies/dsa-vetted-researchers)

**Bluesky (AT Protocol)**
- Bluesky runs official public Jetstream instances "that you can use without authentication". You filter by passing `wantedCollections`, for example `?wantedCollections=app.bsky.feed.post` (replies are `app.bsky.feed.post` records with a `reply` field) — [atproto.com: Streaming data](https://atproto.com/guides/streaming-data); [Jake Lazaroff](https://jakelazaroff.com/words/drinking-from-the-bluesky-firehose/)
- Jetstream cursors are time-based (unix microseconds), so one cursor works across instances — [Bluesky Jetstream docs](https://bsky.network/docs/jetstream)
- Sources disagree on the hostnames. A 2024 commit lists 4 official instances (`jetstream1/2.us-east`, `jetstream1/2.us-west.bsky.network`). The current docs page lists `wss://jetstream.us-east.bsky.network` "for new projects" — [GitHub commit](https://github.com/bluesky-social/jetstream/commit/fccbe56970cccf6d9c528cdd1600947c84ad6772); [bsky.network docs](https://bsky.network/docs/jetstream)
- Jetstream is described as a "full-network archive, replay, and streaming service for atproto", dual-licensed MIT/Apache-2.0, and can be self-hosted — [GitHub: bluesky-social/jetstream](https://github.com/bluesky-social/jetstream)
- Community-run instances also exist (for example `jetstream.fire.hose.cam` and firehose.stream) — [atproto.com](https://atproto.com/guides/streaming-data)

**Mastodon**
- Default limits are **300 calls per 5 minutes per account** and **300 calls per 5 minutes per IP**, reported in `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers. The defaults are hardcoded — [Mastodon docs: Rate limits](https://docs.joinmastodon.org/api/rate-limits/)

**Meta (Facebook, Instagram, Threads) via Meta Content Library (MCL)**
- **Free.** "There are no fees associated with access or computation" — [Meta Transparency: Content Library](https://transparency.meta.com/en-gb/researchtools/meta-content-library/) (page dated 30 Apr 2026)
- **Eligibility:** applicants must be affiliated with a not-for-profit academic or other institution whose primary purpose is scientific or public-interest research. Any discipline can apply and there is no minimum education level. Applications go through Research Tools Manager and are reviewed by CASD (Secure Data Access Center) — [Meta Transparency](https://transparency.meta.com/en-gb/researchtools/meta-content-library/)
- **Coverage:**
  - Facebook: Pages, groups, events, and public profiles that are verified or have 100+ followers.
  - Instagram: business and creator accounts, and personal accounts that are verified or have 100+ followers.
  - Threads: public profiles with 100+ followers, in the web tool only. Threads is not listed among the API endpoints.

  — [Meta Transparency](https://transparency.meta.com/en-gb/researchtools/meta-content-library/)
- **Comments:** the API has comment endpoints for Facebook and Instagram. The page does not mention Threads comments. Bellingcat says comments can be searched within a post — [Meta Transparency](https://transparency.meta.com/en-gb/researchtools/meta-content-library/); [Bellingcat toolkit](https://bellingcat.gitbook.io/toolkit/more/all-tools/meta-content-library)
- **Environment and export:** API analysis must run in the Meta Secure Research Environment or the SOMAR Virtual Data Enclave (a cleanroom). Downloads are limited to posts from widely known individuals and organizations. A query returns at most 100,000 results. You may publish without Meta's approval, but you must give notice — [Meta Transparency](https://transparency.meta.com/en-gb/researchtools/meta-content-library/)
- Meta's page mentions "content owner details" and a data dictionary of more than 100 fields, but does not list the author fields — [Meta Transparency](https://transparency.meta.com/en-gb/researchtools/meta-content-library/)

**TikTok Research API**
- **Rate limit:** 1,000 requests a day, up to 100,000 records a day. The Video and Comments APIs return up to 100 records per request. The quota resets at 00:00 UTC. You can ask for an increase at Research-API@tiktok.com — [TikTok for Developers: Research API FAQ](https://developers.tiktok.com/doc/research-api-faq)
- **Eligibility:** a developer account is not enough. You need to meet eligibility requirements, apply, and be approved for a specific research project — [TikTok FAQ](https://developers.tiktok.com/doc/research-api-faq)
- Third-party sources say eligibility is limited to non-profit academic or public-interest institutions in approved regions (the US, EU, UK and others). I could not confirm the current region list from official pages — [xpoz](https://www.xpoz.ai/blog/guides/tiktok-research-api-limits-access-and-alternatives/)
- The field list is in the Research API codebook — [TikTok codebook](https://developers.tiktok.com/doc/research-api-codebook)
- Researchers have documented data-quality problems with the TikTok Research API, such as missing or inconsistent results — [Cybersecurity for Democracy](https://cybersecurityfordemocracy.org/issues-with-the-tiktok-research-api)

### Inferences
- **Metadata useful for bot detection, by platform.** These are mostly from background knowledge of the API schemas **[unverified this session; check the field references]**:
  - X user objects include `created_at`, `public_metrics` (followers, following, tweet count), `verified` and `description`. Each user lookup costs $0.01.
  - YouTube comment snippets include `authorDisplayName`, `authorChannelId`, `publishedAt`, `updatedAt`, `likeCount` and `textOriginal`. Account age and subscriber count need extra `channels.list` calls at 1 unit each, and many commenter channels hide subscriber counts.
  - Bluesky profiles (`app.bsky.actor.getProfile`) give follower, follow and post counts and `createdAt`. DID creation time can be read from the PLC directory audit log.
  - Mastodon accounts expose `created_at`, `followers_count`, `statuses_count` and `bot` (a self-declared flag).
  - Reddit (PRAW) exposes `created_utc`, `link_karma` and `comment_karma`.
- **X cost example:** at $0.005 per post, 100k replies cost about $500. Adding one $0.01 author lookup per unique author (for example 60k authors) adds about $600. That is well beyond a typical student budget, but the 24-hour dedup window helps if collection is batched by day.
- **YouTube throughput:** 10,000 units a day at up to 100 threads per page is roughly ≤1M top-level comments a day in theory, before channel lookups. In practice it is lower, because of pagination for replies and `channels.list` enrichment.
- **Reddit:** any pipeline built on PRAW now has a short life (it ends by March 2027 unless you are in Reddit for Researchers or have a written agreement). Historical archives (Arctic Shift, old Pushshift dumps) give pre-2023 and 2023+ coverage, but their legal status under Reddit's terms is unclear. For bot-prevalence work, older dumps also predate most LLM-era content, which limits them to a baseline.

### Gaps
- I did not find the official Reddit r/redditdev announcement URL, or the current Reddit for Researchers application page and eligibility (it may have closed or changed).
- I did not verify Bluesky's Developer Guidelines or terms on research use and ML training.
- No official TikTok list of comment fields or approved regions was retrieved. Whether TikTok expanded eligibility in 2026 is unconfirmed.
- I did not check whether the YouTube Data API quota-extension process (an audit form) changed in 2026.
- I did not check the "last updated" dates of the YouTube quota page or the X pricing page. I fetched the X pricing page on 2026-11-11.
- I could not verify Meta's specific author-level fields (account creation date, follower counts) available in MCL.

---

## Q2. Legal, ToS and ethical constraints (scraping vs API, cases, DSA Art. 40, GDPR, IRB, ML-training rules)

### Takeaway
US case law, led by *Meta v. Bright Data*, leans toward allowing logged-off scraping of public data under contract law. But ToS still bind logged-in users and API keyholders, and platforms (Reddit and X especially) are tightening terms. The EU's DSA Article 40 vetted-researcher route has been live since 29 Oct 2025. The first decisions came out in early 2026, but the process is slow (about 80 working days) and needs institutional standing. A student should use official APIs or open protocols and avoid logged-in scraping.

### Cited Findings
- **Meta v. Bright Data (N.D. Cal., Judge Chen, 23 Jan 2024):** the court granted summary judgment to Bright Data on the breach-of-contract claims. Meta's terms "do not apply to and prohibit Bright Data's scraping of publicly available data while logged off". Meta dropped the case on 23 Feb 2024 and did not appeal — [TechCrunch](https://techcrunch.com/2024/02/26/meta-drops-lawsuit-against-web-scraping-firm-bright-data-that-sold-millions-of-instagram-records/); [Pearl Cohen](https://www.pearlcohen.com/court-dismisses-data-scraping-lawsuit-against-bright-data/); [Zyte](https://www.zyte.com/blog/california-court-meta-ruling/)
- Commentators note the analysis is "very fact specific" to Meta's terms and non-logged-in scraping. It may be "a one-off opinion" — [Zyte](https://www.zyte.com/blog/california-court-meta-ruling/); [MediaPost](https://www.mediapost.com/publications/article/392920/bright-data-didnt-violate-metas-terms-by-scrapin.html)
- **X Corp. v. Bright Data (N.D. Cal., 9 May 2024):** the complaint was dismissed (with leave to amend). The court held that X's ToS-based claims against scraping were preempted by copyright law (conflict preemption) — [Proskauer](https://newmedialaw.proskauer.com/2024/05/14/california-court-issues-another-noteworthy-decision-dismissing-breach-of-contract-and-tort-claims-in-web-scraping-dispute/); [Eric Goldman blog / Guy Rub](https://blog.ericgoldman.org/archives/2024/05/x-corp-v-bright-data-is-the-decision-weve-been-waiting-for-guest-blog-post.htm); [Skadden](https://www.skadden.com/-/media/files/publications/2024/05/district-court-adopts/district_court_adopts_broad_view_of_copyright_preemption_in_data_scraping_case.pdf)
- The X v. Bright Data decision was appealed to the Ninth Circuit. I found no Ninth Circuit ruling as of Nov 2026 — [Harvard TagTeam / Goldman blog](https://tagteam.harvard.edu/hub_feeds/3626/feed_items/13124110)
- **X ML-training restriction:** in June 2025, X's developer agreement added that developers "shall not use the X API or X Content to fine-tune or train a foundation or frontier model". X's docs say this is prohibited "with the exception of Grok" — [TechCrunch](https://techcrunch.com/2025/06/05/x-changes-its-terms-to-bar-training-of-ai-models-using-its-content); [X Docs: restricted use cases](https://docs.x.com/developer-terms/restricted-use-cases); [Open Terms Archive](https://opentermsarchive.org/en/memos/x-ban-using-its-content-or-api-to-fine-tune-or-train-large-ai-models/)
- **Reddit:** Rule 8 bars scraping without an authorized agreement. Reddit's data licensing ("other revenue") was $43M in Q2 2026, which shows Reddit is actively monetizing its data — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- **DSA Article 40 Delegated Act:**
  - It was adopted on 2 July 2025 and entered into force on 29 Oct 2025.
  - It covers vetted-researcher access to **non-public** data (Art. 40(4)) and creates the DSA Data Access Portal. Public-data access under Art. 40(12) is outside its scope.
  - Researchers must define and justify their requests (Art. 7) and publish results free of charge.

  — [UKE (Polish DSC)](https://uke.gov.pl/en/newsroom/delegated-act-on-data-access-for-researchers-enters-into-force,423.html); [Hogan Lovells](https://www.hoganlovells.com/en/publications/who-gets-to-see-inside-the-eus-new-rules-on-data-access-under-article-40-of-the-dsa)
- Applications for vetted status opened on 29 Oct 2025 via the portal (data-access.dsa.ec.europa.eu). The Digital Services Coordinator (DSC) grants vetted status and requests the data from the very large platform (VLOP) — [ACM (Netherlands)](https://www.acm.nl/en/publications/access-platform-data-vetted-researchers-under-dsa)
- Ireland's Coimisiún na Meán FAQ appears to describe a one-step process, which conflicts with the ACM's two-step description — [CnaM FAQ PDF](https://www.cnam.ie/app/uploads/2025/12/DSA-Article-40-Data-Access_-Frequently-Asked-Questions.pdf); [CnaM](https://www.cnam.ie/industry-and-professionals/online-safety-framework/certifications-schemes/vetted-researchers/)
- First-wave decisions began arriving in early 2026. A DSA Observatory author described a **rejected** request: the DSC notified them after about 80 working days, and the reason was a failure to prove vetted-researcher status — [DSA Observatory, 12 Mar 2026](https://dsa-observatory.eu/2026/03/12/if-at-first-you-dont-succeed-reflections-on-a-rejected-art-40-dsa-data-access-request/)
- On 19 May 2026 the Commission and DSCs held a roundtable with VLOPs/VLOSEs to support the launch of the first data access requests — [European Commission](https://digital-strategy.ec.europa.eu/en/news/commission-holds-roundtable-data-access-vetted-researchers)
- Tech Policy Press discusses what the Commission's fine on X shows about Art. 40 data-access enforcement — [Tech Policy Press](https://techpolicy.press/what-the-x-fine-reveals-about-data-access-under-article-40-of-the-digital-services-act)
- Meta Content Library lets researchers publish without Meta's prior approval, but requires attribution and notice — [Meta Transparency](https://transparency.meta.com/en-gb/researchtools/meta-content-library/)

### Inferences
- **hiQ v. LinkedIn** **[unverified this session]**: the Ninth Circuit (2019, reaffirmed 2022 after *Van Buren*) held that scraping public pages likely does not violate the CFAA ("without authorization"). The district court later found hiQ liable for breach of LinkedIn's User Agreement (Nov 2022), and the case settled. The lesson is that CFAA risk is low for public data, but **contract/ToS liability remains**, especially for logged-in accounts and fake accounts.
- **Practical legal posture for a student:**
  1. Use official APIs or open protocols (Bluesky, Mastodon, YouTube).
  2. Never scrape while logged in, never use fake accounts, and never bypass technical barriers (CAPTCHAs, rate limits).
  3. Respect robots.txt and rate limits.
  4. Don't use the data to train foundation models, which X prohibits outright. A small research classifier is arguably not a "foundation or frontier model", but check each platform's terms.
- **GDPR / privacy** **[unverified this session; the general framework is well established]**:
  - Usernames, handles and comment text are personal data under GDPR when they relate to identifiable people, even if public.
  - Research processing usually relies on Art. 6(1)(e) or (f) plus the Art. 89 research safeguards (pseudonymization, data minimization).
  - Recommended handling:
    - Hash or pseudonymize author IDs at ingest (keyed HMAC, with the key stored separately).
    - Do not publish verbatim comments that can be searched back to a person.
    - Keep a data management plan and retention schedule.
    - Honor deletions (X and YouTube terms require removing deleted content).
  - YouTube's API Developer Policies require refreshing or deleting stored API data periodically (historically within 30 days) **[unverified this session]**.
- **IRB/ethics** **[unverified this session]**:
  - Collecting public posts without interaction is often deemed "not human subjects research" or exempt under US Common Rule category 4. The student's IRB must still make that determination.
  - Follow the AoIR Internet Research Ethics 3.0 guidelines.
  - Labeling accounts as "bots" or "AI" creates a risk of mislabeling real people. Report results in aggregate, and don't name individual accounts.
- **Sharing data:** share IDs (post or comment IDs) rather than full content. This is the common norm on X and Reddit, and it lets others "rehydrate" while respecting deletions.

### Gaps
- No primary-source check of hiQ v. LinkedIn this session, and no check of the current Ninth Circuit docket in X v. Bright Data.
- I did not count how many Art. 40 requests have been approved in 2026 (none found), or whether students or unaffiliated researchers can qualify (it likely needs a research-organization affiliation under Art. 40(8)).
- I did not research Bluesky, Mastodon-instance, YouTube or TikTok rules on using data to train ML models.
- I did not research GDPR guidance from specific DPAs or EDPB statements on research scraping, nor the AI Act's interaction with this.
- I did not verify YouTube API Services Developer Policies data-retention clauses for 2026.

---

## Q3. Tooling, storage schema, deduplication and sampling strategy

### Takeaway
The workable stack is Python with platform-native clients (google-api-python-client, atproto or a websocket client for Jetstream, Mastodon.py, and PRAW only until March 2027). Store raw JSON plus a normalized table in SQLite/DuckDB or Postgres, keyed by `(platform, comment_id)`. Sample with a pre-registered, stratified design rather than convenience "hot" threads.

### Cited Findings
- Jetstream lets you subscribe to only `app.bsky.feed.post` (and repeat the parameter for more collections). Its time-based cursors in unix microseconds enable resumable, replayable collection — [atproto.com: Streaming data](https://atproto.com/guides/streaming-data); [Bluesky docs](https://bsky.network/docs/jetstream)
- Mastodon's rate-limit headers (`X-RateLimit-Remaining` and `X-RateLimit-Reset`) let a client throttle itself adaptively — [Mastodon docs](https://docs.joinmastodon.org/api/rate-limits/)
- X deduplicates charges for resources requested again within the same UTC day, so batching re-fetches within a day avoids double billing — [X Docs: Pricing](https://docs.x.com/x-api/getting-started/pricing)
- Reddit's Devvit replacement has no Python support. PRAW-based tools need migration or a commercial agreement — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- A Cybersecurity for Democracy open-source library exists for the TikTok Research API — [C4D](https://cybersecurityfordemocracy.org/streamlining-tiktok-research)

### Inferences
- **Libraries** **[names and roles from background knowledge; check current versions]**:

  | Library | Use | Notes |
  |---|---|---|
  | `google-api-python-client` | YouTube (`commentThreads().list(part="snippet,replies", videoId=..., maxResults=100, textFormat="plainText")`) | |
  | `atproto` (MarshalX) | Bluesky profiles, threads, search (`app.bsky.feed.getPostThread`) | Use `websockets` or `aiohttp` for Jetstream JSON streams |
  | `Mastodon.py` | Mastodon public and hashtag timelines, `status_context` for replies | |
  | `PRAW` | Reddit | Only until shutdown |
  | `yt-dlp --write-comments --skip-download` | YouTube comments without quota | Scrapes the web interface rather than the API, so it sits outside YouTube's API terms and the YouTube ToS restricts automated access. Treat it as a ToS risk and use it only for small validation checks |
  | `tweepy` / `xdk` | X | Pay-per-use cost applies |

- **Storage schema** (one normalized table plus a raw store):
  - `comments(platform, comment_id PK, parent_id, root_post_id, thread_depth, author_id_hash, created_at_utc, text, lang, like_count, reply_count, edited_at, collected_at, source_endpoint, sample_stratum, raw_json_ref)`
  - `authors(platform, author_id_hash PK, account_created_at, followers, following, post_count, self_declared_bot, display_name_hash, snapshot_at)`. Take author snapshots on a schedule, because counts change over time.
  - `posts(platform, post_id PK, channel_or_community, topic_stratum, created_at, view_count)`
  - Keep the raw API JSON (compressed JSONL or Parquet) so it can be re-parsed later. DuckDB over Parquet works well for a solo developer.
- **Deduplication:**
  - Use a primary key on `(platform, native_id)` with upsert.
  - For text near-duplicates (copy-paste spam, which is itself a bot signal), store a normalized-text hash (lowercase, URL- and emoji-stripped) and a MinHash/SimHash signature. Flag near-duplicates rather than deleting them, since duplication is part of what you're measuring.
  - Deduplicate across Jetstream instances and replays by AT-URI (`at://did/app.bsky.feed.post/rkey`).
- **Sampling strategy** (to avoid bias):
  1. Define the population explicitly, for example "comments on English-language YouTube videos from channels with more than 10k subscribers in categories X/Y/Z, posted in a given window".
  2. Stratify by topic (politics, crypto/finance, entertainment, tech, health), by channel/community size tiers, and by post age. Bot prevalence is known to vary sharply by topic, especially crypto and political content.
  3. Within strata, randomly sample posts. Don't use "trending" or "hot" lists only, since they over-represent viral threads where engagement bots cluster.
  4. Collect **all** comments on each sampled post, or a fixed random fraction, including late replies. Re-crawl at fixed intervals (24h, 7d) to capture later bot waves and moderation removals.
  5. For streams (Bluesky Jetstream), time-sample (for example random 1-hour windows spread over weeks) to avoid day-of-week and news-cycle bias.
  6. Record the sampling frame and seeds for reproducibility, and pre-register prevalence estimands.
  7. Survivorship bias: platforms remove spam before you see it, so measured prevalence is a *lower bound* of what was posted. Collecting near-real-time (Jetstream) versus after the fact (YouTube API) gives different numbers. Note this in methods.

### Gaps
- I did not confirm current versions or maintenance status of PRAW, atproto, Mastodon.py or tweepy in 2026, nor whether X's official Python SDK ("xdk") is the recommended client.
- I did not find published 2025-2026 sampling-methodology papers specific to bot-prevalence estimation. Other researchers may cover this.

---

## Q4. Recommended starter platform(s) for a low-budget project

### Takeaway
Start with **Bluesky (Jetstream plus the AppView API)** as the primary source: free, unauthenticated, full-network, and replayable, with author metadata available. Add **YouTube Data API v3** as the second source: free within 10k units a day, with large comment volumes and a known AI-spam problem. Optionally add **Mastodon** for a low-bot contrast. Skip Reddit (API closing), X (pay-per-use costs and ML-training ban), Meta and TikTok (institutional vetting, secure-enclave export limits) unless a supervisor or institution can sponsor access.

### Cited Findings
- Bluesky Jetstream: official instances usable without authentication, with collection filtering and time-based cursors — [atproto.com](https://atproto.com/guides/streaming-data); [Bluesky docs](https://bsky.network/docs/jetstream)
- YouTube: 1 unit per `commentThreads.list` call, and 10,000 units a day free by default — [Google](https://developers.google.com/youtube/v3/determine_quota_cost)
- Mastodon: 300 requests per 5 minutes per account or IP by default — [Mastodon docs](https://docs.joinmastodon.org/api/rate-limits/)
- X: $0.005 per post read and $0.01 per user read, 3M reads a month cap, no free tier for new signups — [X Docs](https://docs.x.com/x-api/getting-started/pricing); [Open Terms Archive](https://opentermsarchive.org/en/memos/x-replaces-its-dollar200-entry-level-api-plan-with-a-pay-per-use-model/)
- Reddit: no new public API requests after 31 Oct 2026, with the shutdown finishing in March 2027 — [PPC Land](https://ppc.land/reddit-to-halt-public-data-api-access-in-march-2027-as-14-000-apps-register/)
- Meta Content Library and TikTok Research API: free, but only for vetted, institution-affiliated researchers with an approved project — [Meta](https://transparency.meta.com/en-gb/researchtools/meta-content-library/); [TikTok FAQ](https://developers.tiktok.com/doc/research-api-faq)

### Inferences
- **Suggested pipeline for a solo student:**
  1. Run a Bluesky Jetstream consumer (asyncio websocket, `wantedCollections=app.bsky.feed.post`, persist the cursor) that writes replies (posts with a `reply` field) to Parquet.
  2. Run a nightly enrichment job that batch-fetches author profiles for new DIDs via `app.bsky.actor.getProfiles` (up to 25 per call), plus DID creation dates.
  3. Run a YouTube job: pick a stratified sample of videos (via `search.list`, sparingly, since it has its own small bucket, or from curated channel lists), pull `commentThreads` with replies, then do `channels.list` enrichment for commenter channels, all within 10k units a day.
  4. Load everything into DuckDB with the schema above and deduplicate.
  5. Pseudonymize author IDs at ingest, and keep a mapping key offline.
  6. Get an IRB determination before starting collection.
- **If the student is at an EU institution:** consider filing a DSA Art. 40 request through a supervisor for X, TikTok or Meta data as a longer-term (more than 3 months) parallel track. Expect about 80 working days and possible rejection over vetted-status proof, per the DSA Observatory.
- **If Reddit is essential:** apply to Reddit for Researchers through a PI, or use historical archives only for a pre-LLM baseline, noting the legal ambiguity.
- **Bias caveat for the write-up:** Bluesky's user base and moderation differ from mainstream platforms, so prevalence there does not generalize. Using two platforms (Bluesky plus YouTube) lets you compare across platforms.

### Gaps
- I found no published 2025-2026 measurement of AI-comment prevalence on Bluesky versus YouTube to support the claim that these platforms are useful for bot measurement. That belongs to adjacent research.
- I did not verify Bluesky's current ToS or Developer Guidelines on bulk collection and research use. Check them before publishing.
- The pricing and terms above were confirmed on 2026-11-11 from the cited pages. X explicitly says "prices are subject to change", and the Reddit timelines come from secondary reporting of Reddit's 30 Sep 2026 posts.

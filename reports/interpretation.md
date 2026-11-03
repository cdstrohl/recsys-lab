*Draft by the coding agent for owner review. Written against the manifest `v1` results;
re-read it whenever the tables change.*

### Strongest simple baseline

On holdout, `category_affinity` is the strongest simple ranker. It beats `popularity`
and `tfidf` overall, in every bucket of users with History, and on both sports slices,
and the paired intervals against both exclude zero. `tfidf` is second and also
clearly beats `popularity`. `random` sits below everything, as a sanity floor should.

**Caveat: the tune split disagrees.** On tune, `tfidf` led `category_affinity`
overall and in every bucket of users with History, though in the shortest-History
bucket that lead's interval includes zero. Had we picked a model on tune, we would have
picked `tfidf`. Two things differ between the splits: the day, and the statistics
source (fit only for tune, fit + tune for holdout). Until a tune-only diagnostic
explains the flip, the order of the two content rankers should be treated as
unstable from day to day. Their lead over `popularity` holds on both splits.

### What I'd ship first, and why

Ship `category_affinity` with the explicit `popularity` fallback:
- **Quality:** it is the best-scoring model on holdout.
- **Cost:** it is the cheapest personalized model. It counts categories in the
  user's History, needs no text model or vocabulary, and scores faster than `tfidf`
  at both the median and the tail.
- **Explainability:** every recommendation has a one-line reason ("because you read
  sports / NFL"), which suits a reason-code demo.
- **Cold start is explicit.** Users with no History get the popularity order, and
  the fallback rate is measured.

Treat it as the control for the next ladder step (item-item CF and a hybrid), not as
a final answer. Given the tune/holdout flip, a hybrid that uses both category and
text signals is the natural next candidate.

### Patterns worth knowing

- **Personalization pays more as History grows.** The gap between the personalized
  rankers and `popularity` widens from short to long histories.
- **Compare models within a bucket, not across buckets.** `random` also scores
  lower for long-History users, so their impressions are harder to rank. A falling
  line doesn't mean a model got worse for heavy readers.
- **Zero-History users get nothing beyond popularity here.** That's by construction
  (fallback). Offline, popularity itself is weak, because dev's fresh items have no
  train clicks and dev clicks are never features. A production feed would use
  minutes-old popularity, which this protocol deliberately can't simulate.
- **Coverage:** the personalized rankers put far more distinct items into a top ten
  than `popularity`, which concentrates on a few already-clicked items.
- **The sports slices don't say much yet.** Nearly every impression has at least one
  sports candidate, so that slice is close to the whole split. The "sports History"
  slice shows the same model order as the overall table. Nothing here is evidence of
  sports-specific behavior.

### What offline evaluation can't predict

- **Exposure bias:** labels come from what Microsoft News's 2019 system chose to
  show, so a ranker that would surface different items is judged only on items that
  policy already picked.
- **Position and presentation effects:** layout, images, reason codes and novelty
  wear-off are invisible in logged candidate lists.
- **Long-run effects:** feed diversity, filter bubbles, return visits and retention
  aren't measured by a per-impression ranking metric.
- **Freshness:** a live system's popularity signal is minutes old, not a day old.

None of these numbers are CTR or engagement. Deciding whether `category_affinity`
helps real users needs an online experiment: user-level randomization, a primary
metric chosen in advance, guardrails, and exposure logging.

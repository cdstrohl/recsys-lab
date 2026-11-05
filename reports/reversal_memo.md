# Research memo: the tune/holdout reversal, and what to test next

*Hand-written by the coding agent for owner review, 2026-11-05. All numbers are taken
from [baseline_diagnostics.md](baseline_diagnostics.md) (generated) and DECISIONS.md
D-011/D-021. Offline ranking metrics only; nothing here is engagement or CTR.*

## Question

On tune (11-14), `tfidf` beat `category_affinity` by +0.014 nDCG@10. On holdout (11-15)
it lost by −0.027. That is a swing of −0.040. Did the swing come from the training
corpus (holdout's rankers were refitted on one more day) or from the evaluation
population? The holdout was already inspected, so its numbers are used here only
descriptively. Everything new was measured on train days.

## Findings

1. **Corpus size explains almost none of it.** With the tune impressions held fixed,
   I varied the statistics the rankers are fitted on from 25% of `fit` up to all of
   it, and from 1 to 5 trailing days. The gap stayed between +0.0125 and +0.0141. The
   largest move toward holdout is 3% of the swing. Popularity's lead over `random` is
   also flat, except with a 1-day window, where recency helps it slightly.
2. **The measurable population shifts explain almost none of it either.**
   - **History-length mix:** tune and holdout have nearly the same History-length mix.
     Reweighting tune to holdout's buckets moves the gap by 0% of the swing.
   - **Returning vs new users:** this was my leading hypothesis. 88% of holdout users
     are new to train, against 16% of tune impressions. Within each train day, though,
     returning and new users show nearly the same gap, so reweighting accounts for
     only 4% of the swing.
   - **Concentration mix:** History concentration is stable from day to day.
3. **The day itself is what moves.** With the corpus held at a one-day window, the gap
   by scored day is −0.0063, −0.0123, +0.0038 and +0.0125. That's the same pattern as
   with all earlier data. The shift happens inside population slices at the same time:
   - every History-length bucket with History
   - every concentration bucket except the most concentrated (top category over two
     thirds, about 4% of impressions), which shows no clear trend
   - both returning and new users

   Something about each day's impressions moves the gap: which stories are in the
   candidate lists, and which ones people clicked. The user mix doesn't.
4. **Holdout is more extreme than any train day.** Its gap (−0.027) is below the
   lowest train day (−0.0125). The day-to-day variation in train already covers both
   signs. So the holdout result isn't anomalous in direction, only in size.
5. **`category_affinity` mostly hands its decisions to popularity.**
   - About 80–83% of clicked candidates share their affinity score with at least one
     unclicked candidate, so the popularity tie-break decides their order.
   - For `tfidf` it is 15–17%.
   - Taken with D-021, where random tie-breaks barely moved the gap, this says
     `category_affinity` is a coarse category filter. Its quality depends on whether
     the day's clicks fall in the user's habitual categories.
6. **Concentration matters within a day, in a consistent direction.** On tune, `tfidf`'s
   lead is largest for users with diverse History (+0.024 when the top category is
   under a third of History). It shrinks as concentration rises, and turns negative
   (−0.012, interval includes 0) above two thirds. The ordering across the three lower
   concentration buckets holds on all four train days.
7. **Coverage isn't the bottleneck.** Every candidate gets a non-empty TF-IDF vector.
   21–29% of candidates fall outside the corpus the vocabulary was fitted on, but are
   still represented. 54–60% of candidates were never clicked in the statistics
   source, which is why the popularity tie-break is often uninformative.

**Methodological lesson.** Our 95% intervals resample users within one day. They don't
cover day-to-day variation, and here day-to-day variation is the dominant source of
uncertainty: the tune interval excluded zero, and the sign still flipped the next day.
Any claim of the form "model A beats model B" needs consistency across days, not only
a user-bootstrap interval.

**Unexpected, and not pre-registered.** Mean candidate-list length rose from 35.6 on
11-11 to 40.4 on 11-14. The two days with longer lists are the two that favored
`tfidf`. That's four points, so treat it as a lead, not a finding.

## Two follow-up experiments

**E1. Within-category discrimination (diagnostic, train days only).**
- **Hypothesis:** `tfidf` beats `category_affinity` on impressions where many
  candidates share the user's top affinity score, because only `tfidf` can rank inside
  a category.
- **Measure:** per impression, the share of candidates tied at the maximum affinity
  score. This is label-free.
- **Fix before running:**
  - the bucket edges
  - prediction 1: the gap rises across buckets on at least three of four days
  - prediction 2: each day's mix of this measure, together with list length, tracks
    the daily gap
- **If it holds:** the day effect has an item-side mechanism, and the hybrid in E2
  should get this measure as an input.
- **Cost:** no new model, about 10 minutes of compute.

**E2. Leave-one-day-out hybrid (the D-022 protocol, narrowed).**
- **Model:** logistic regression over the out-of-time scores of `category_affinity`
  and `tfidf`, the popularity keys, and one pre-specified interaction: `tfidf` × the
  user's top-category share, motivated by finding 6.
- **Training:** fitted on the four rolling development days (D-022 folds), with
  leave-one-day-out for the regularization strength.
- **Success:** both conditions must hold:
  - it beats both base rankers on at least three of four held-out days
  - its pooled paired interval against `category_affinity` is above 0
- **Ablations:**
  - without the interaction
  - with the five D-021 candidates and `popularity_48h` as extra inputs, each kept
    only if it helps across days
- **Reporting:** per-day results are reported beside the aggregate.

## A fresh evaluation protocol

1. **Develop on train days only.** Every choice comes from the four rolling-origin
   folds:
   - models
   - features
   - hyperparameters
   - E1/E2 decisions

   Report day-level consistency next to the user-bootstrap intervals.
2. **Freeze before looking, in a new manifest `v2`.** The `v1` manifest isn't edited.
   Freeze:
   - the code commit and model list (including `popularity_48h`)
   - the primary metric (nDCG@10)
   - the primary comparison: the hybrid vs `category_affinity`, with the spec's 5%
     relative threshold
   - the secondary comparisons and the analysis plan
3. **Label any MIND-small dev result as exploratory.** It's already been seen. It may
   still be re-scored for continuity with `v1`, under the label *exploratory holdout*,
   and it can't change a frozen decision.
4. **Run one confirmatory evaluation on data nobody has opened.** Two options, both
   needing owner approval:
   - **MIND-large dev, users disjoint from MIND-small.** These are new users on the
     already-seen day (11-15), so it tests generalization across users, not across
     days. It needs a license review, a manual download, and a check by ID that the
     users really are disjoint.
   - **REC-10, the MIND-large test leaderboard.** It covers about a week of unseen
     days, so it's the only option that tests robustness across days, which this
     analysis shows matters most. It's scored externally as one aggregate, isn't
     comparable to MIND-small numbers, and an owner-approved submission is required.

   I recommend the first as the confirmatory test for MIND-small-scale claims, and the
   second if the portfolio needs an out-of-time result.
5. **Report the confirmatory result once, whatever it shows.** Nothing is re-tuned
   afterwards, and a null result is published as a null result.

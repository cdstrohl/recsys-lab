# Data: Microsoft MIND-small

This repo ships **no MIND data** and **no download code**. You download the files
by hand after reading and accepting the license. Everything under `data/raw/` and
`data/archives/`, and every `*.zip`, is gitignored.

## License

MIND is released by Microsoft under the **Microsoft Research License Terms**.
Read the terms on the official site before downloading. This repo doesn't restate
or interpret them. Don't redistribute the files, commit them, or show raw records
or article text in public demos.

Citation:

> Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu,
> Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. 2020. *MIND: A Large-scale Dataset
> for News Recommendation.* In Proceedings of ACL 2020.

## What to download

The download links on <https://msnews.github.io/> point to a gated Hugging Face
dataset, [`yjw1029/MIND`](https://huggingface.co/datasets/yjw1029/MIND). Before the
links work:

1. Sign in to Hugging Face (a free account is enough).
2. Open the dataset page and accept its access conditions (the Microsoft Research
   License Terms). Approval is automatic.
3. Go back to the download links, or use the dataset's "Files and versions" tab.

The site lists two groups. The first group (Training, Validation and Test Set) is
MIND-large. Use only the **MIND-small** group further down:

| Archive | Site label | Used as |
|---|---|---|
| `MINDsmall_train.zip` | MIND-small Training Set | `fit` and `tune` splits |
| `MINDsmall_dev.zip` | MIND-small Validation Set | `holdout` split (locked) |

**Don't** use the `MINDlarge_*` files, the MIND-large "Test Set" (its labels aren't
public), or third-party mirrors or reformatted CSVs. Their schemas differ and they're
out of scope.

## Where to put it

```bash
mkdir -p data/archives data/raw
# Move the two downloaded zips into data/archives/, then:
unzip data/archives/MINDsmall_train.zip -d data/raw/MINDsmall_train
unzip data/archives/MINDsmall_dev.zip   -d data/raw/MINDsmall_dev
```

Some browsers unzip downloads automatically. Safari does by default and deletes the
zip. If you got folders instead of zips, move their contents straight into
`data/raw/MINDsmall_train/` and `data/raw/MINDsmall_dev/`.

Expected layout. The code reads only `news.tsv` and `behaviors.tsv`. The archives
also contain entity and relation embedding files, which aren't used.

```
data/raw/
  MINDsmall_train/{news.tsv, behaviors.tsv, ...}
  MINDsmall_dev/{news.tsv, behaviors.tsv, ...}
```

If an archive unpacks into an extra nested folder, move the files up so the paths
match the layout above.

## Record the hashes

Run these from the repo root. They record SHA-256 hashes of the four TSVs the code
reads:

```bash
shasum -a 256 data/raw/MINDsmall_*/news.tsv data/raw/MINDsmall_*/behaviors.tsv \
  > data/archives/SHA256SUMS
cat data/archives/SHA256SUMS                   # four lines: hash, two spaces, path
shasum -a 256 -c data/archives/SHA256SUMS      # later: every line should say OK
```

The TSV hashes are what count, because they cover exactly what the code reads. If
you also kept the zips, add `data/archives/MINDsmall_*.zip` to the first command to
record which download you used.

`SHA256SUMS` stays local, since `data/archives/` is gitignored. The hashes reveal
nothing about the data. They are copied into `eval/manifest.json`, which is locked and
committed before the first holdout run (REC-07). If `-c` reports `FAILED`, the files
changed since you recorded them, so stop and find out why before running any
evaluation.

## Check it

```bash
uv run python scripts/inspect_data.py --data data/raw
```

This prints aggregate counts only: rows, date range, impressions per day, no-click
and all-click impressions, history-length buckets, malformed rows, and the share of
dev candidates unseen in train. Add `--json path.json` to save them.

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

From <https://msnews.github.io/> (check the file names on the download page):

| Archive | Used as |
|---|---|
| `MINDsmall_train.zip` | `fit` and `tune` splits |
| `MINDsmall_dev.zip` | `holdout` split (locked) |

**Don't** use the MIND-large files, the MIND-large "Test Set" (its labels aren't
public), or third-party mirrors or reformatted CSVs. Their schemas differ and they're
out of scope.

## Where to put it

```bash
mkdir -p data/archives data/raw
# Move the two downloaded zips into data/archives/, then:
shasum -a 256 data/archives/MINDsmall_*.zip    # keep these hashes for eval/manifest.json
unzip data/archives/MINDsmall_train.zip -d data/raw/MINDsmall_train
unzip data/archives/MINDsmall_dev.zip   -d data/raw/MINDsmall_dev
```

Expected layout. The code reads only `news.tsv` and `behaviors.tsv`. The archives
also contain entity and relation embedding files, which aren't used.

```
data/raw/
  MINDsmall_train/{news.tsv, behaviors.tsv, ...}
  MINDsmall_dev/{news.tsv, behaviors.tsv, ...}
```

If an archive unpacks into an extra nested folder, move the files up so the paths
match the layout above.

## Check it

```bash
uv run python scripts/inspect_data.py --data data/raw
```

This prints aggregate counts only: rows, date range, impressions per day, no-click
and all-click impressions, history-length buckets, malformed rows, and the share of
dev candidates unseen in train. Add `--json path.json` to save them.

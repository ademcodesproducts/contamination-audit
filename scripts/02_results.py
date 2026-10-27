"""Regenerate every number reported in paper/paper.tex.

Each stage writes one JSON file under results/paper/ and prints its figures the
way the paper states them, so the two can be diffed. Stages are independent and
share no state; --stage runs one, no argument runs all twelve in this order:

  main         Table 2, flag counts per filter on both benchmark groups
  strata       the control set's two genres, scored separately
  consensus    Sec. 6, filter agreement and bootstrap intervals
  scaling      Fig. 1, error fraction against corpus size      (slowest, ~25 min)
  tuning       Fig. 2, the n-gram and threshold sweeps
  mechanism    Sec. 9, which n-grams the false positives match on
  composition  Sec. 11, what kind of text the flagged items are
  validate     Sec. 5, tests the control assumption itself
  second       Sec. 12, replication on a second corpus
  matched      Sec. 13, same competition one year apart
  nemotron     Sec. 14, third corpus, with its own control split
  phrases      Sec. 15, Table 5, the phrase-agreement sweep

decon is not run here: it is a separate Rust binary. See scripts/03_run_decon.py.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import logging
import random
from pathlib import Path

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.filters import (
    flag_by_fuzz,
    flag_by_ngram,
    flag_by_sampled_phrases,
    ngrams,
    registry,
    tok_qwen,
    tok_s1,
)
from contamination_audit.io import load_jsonl

_log = logging.getLogger("results")

OUT = Path("results/paper")
CORPUS = "data/raw/dolci_questions.jsonl"
MAIN_N = 100_000
SEED = 42
BOOTSTRAP = 10_000

# The sweep grids the paper reports. Each is a parameter some project leaves
# unstated, which is why they are swept rather than fixed.
SCALES = [12_500, 25_000, 50_000, 100_000, 150_000, 250_000, 390_000]
NGRAM_SIZES = [5, 8, 10, 13, 15, 20, 25, 30]
FUZZ_THRESHOLDS = [95, 90, 85, 80, 75, 70, 65, 60]
PHRASE_FRACTIONS = [0.50, 0.30, 0.20, 0.10, 0.05]

# Longer than NGRAM_SIZES: this one asks how long a shared span gets, so it has
# to reach past where the filters stop flagging anything.
SPAN_SIZES = [8, 10, 13, 15, 20, 25, 30, 40]

# Benchmarks released before Nemotron-SFT-Math-v4 (2026-05-01) but after Dolci.
# They are controls for Dolci and measurement candidates for Nemotron: the control
# set is a function of the corpus, not a fixed list.
PRE_MAY_2026 = {
    "aime_2026",
    "hmmt_feb_2026",
    "usamo_2026",
    "arxivmath-0126",
    "arxivmath-0326",
    "brokenarxiv-0326",
}


# ---------------------------------------------------------------- loading


def corpus(n: int) -> list[str]:
    """The first `n` questions of the corpus.

    Always the first n, never a sample: every scaling point is a prefix of the
    next, so the curve measures corpus size and nothing else.
    """
    questions = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            questions.append(json.loads(line)["problem"])
    # A short file would otherwise be reported under the size that was asked
    # for, labelling a 100k run as 390k.
    if len(questions) != n:
        raise SystemExit(f"{CORPUS} holds {len(questions)} questions, needed {n}")
    return questions


def benchmarks() -> tuple[list[dict], list[dict]]:
    """The measurement and control groups, as records with id and benchmark."""
    olymmath = [
        {"id": r["id"], "benchmark": "olymmath", "problem": r["problem"]}
        for r in load_jsonl("data/raw/olymmath_en_hard.jsonl")
        + load_jsonl("data/raw/olymmath_en_easy.jsonl")
    ]
    measure = olymmath + load_jsonl("data/raw/bench_measure.jsonl")
    control = load_jsonl("data/raw/bench_control.jsonl")
    return measure, control


def benchmark_texts() -> tuple[list[str], list[str]]:
    """Just the problem statements -- what a filter actually reads."""
    measure, control = benchmarks()
    return [r["problem"] for r in measure], [r["problem"] for r in control]


def filters():
    """The four faithful filters, in the order the paper's tables use."""
    return [(f.name, f.provenance, f.run) for f in registry()]


def load_decon():
    """What scripts/03_run_decon.py recorded, or None if it was never run.

    decon is a Rust binary and is not part of the filter registry, so its counts
    come from that script's output rather than from a run here.
    """
    path = OUT / "decon.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    _log.info("wrote %s", OUT / f"{name}.json")


# ------------------------------------------------------------- statistics


def percentile_interval(draws) -> tuple[float, float]:
    """The central 95% of a bootstrap distribution."""
    ordered = sorted(draws)
    return (
        ordered[int(0.025 * len(ordered))],
        ordered[min(int(0.975 * len(ordered)), len(ordered) - 1)],
    )


def rate_with_interval(flagged: set[int], total: int) -> dict:
    """A flag rate and its 95% bootstrap interval.

    Resamples the observed indicator vector rather than simulating a binomial:
    nonparametric, and it assumes nothing about the shape of the distribution.
    """
    indicator = [1 if i in flagged else 0 for i in range(total)]
    rng = random.Random(SEED)
    draws = sorted(
        100 * sum(rng.choices(indicator, k=total)) / total for _ in range(BOOTSTRAP)
    )
    return {
        "pct": 100 * len(flagged) / total,
        "ci": [draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP)]],
    }


# ---------------------------------------------------------------- stages


def stage_main():
    """Table 2: flag counts per filter on both benchmark groups."""
    docs = corpus(MAIN_N)
    measure_texts, control_texts = benchmark_texts()
    n_measure, n_control = len(measure_texts), len(control_texts)

    print(
        f"\nTable 2  (corpus={len(docs)}, measurement={n_measure}, control={n_control})"
    )
    print(f"{'filter':24s} {'params':6s} {'measurement':>14s} {'control':>14s}")

    results = {}
    for name, provenance, run in filters():
        flagged_measure = run(docs, measure_texts)
        flagged_control = run(docs, control_texts)
        results[name] = {
            "provenance": provenance,
            "measure": sorted(flagged_measure),
            "control": sorted(flagged_control),
        }
        print(
            f"{name:24s} {provenance:6s} "
            f"{len(flagged_measure):5d} ({100 * len(flagged_measure) / n_measure:5.1f}%) "
            f"{len(flagged_control):5d} ({100 * len(flagged_control) / n_control:5.1f}%)"
        )

    # decon is the fifth row of Table 2, but it is a Rust binary rather than a
    # filter in this registry, so scripts/03_run_decon.py runs it separately.
    # Read what that run actually reported; do not assume it found nothing.
    decon = load_decon()
    if decon is None:
        print(f"{'decon':24s} {'code':6s} {'NOT RUN -- see 03_run_decon.py':>31s}")
    else:
        print(
            f"{'decon (see 03_run_decon)':24s} {'code':6s} "
            f"{decon['measure']:5d} ({100 * decon['measure'] / n_measure:5.1f}%) "
            f"{decon['control']:5d} ({100 * decon['control'] / n_control:5.1f}%)"
        )

    save(
        "main",
        {
            "corpus_n": len(docs),
            "n_measure": n_measure,
            "n_control": n_control,
            "filters": results,
        },
    )


def stage_consensus():
    """Section 6: how many filters agree on each flagged item."""
    main_results = json.loads((OUT / "main.json").read_text(encoding="utf-8"))
    n_measure, n_control = main_results["n_measure"], main_results["n_control"]

    flagged_by = {
        name: set(r["measure"]) for name, r in main_results["filters"].items()
    }

    # decon is one of the five filters compared, so it belongs in the denominator
    # of "how many filters agree". scripts/03_run_decon.py records only counts,
    # not which items it hit, so an empty set is right precisely when that count
    # is zero. If it ever hits something, the agreement numbers below cannot be
    # computed without the indices -- stop rather than silently understate them.
    decon = load_decon()
    if decon is None:
        raise SystemExit(
            "consensus needs decon's result for the agreement denominator; "
            "run: python scripts/03_run_decon.py --decon-bin ..."
        )
    if decon["measure"]:
        raise SystemExit(
            f"decon flagged {decon['measure']} measurement items, but decon.json "
            "records counts only. Extend scripts/03_run_decon.py to save the "
            "flagged indices before computing agreement."
        )
    flagged_by["decon"] = set()
    n_filters = len(flagged_by)

    def agreement_counts(indices):
        """For each k >= 1, how many of these items exactly k filters flag."""
        counts = collections.Counter()
        for i in indices:
            k = sum(i in flagged for flagged in flagged_by.values())
            if k:
                counts[k] += 1
        return counts, sum(counts.values())

    observed, n_flagged = agreement_counts(range(n_measure))

    rng = random.Random(SEED)
    resampled = {k: [] for k in range(1, n_filters + 1)}
    for _ in range(BOOTSTRAP):
        indices = [rng.randrange(n_measure) for _ in range(n_measure)]
        counts, total = agreement_counts(indices)
        for k in range(1, n_filters + 1):
            resampled[k].append(counts.get(k, 0) / total if total else 0.0)

    print(
        f"\nSection 6  ({n_flagged}/{n_measure} items flagged by >=1 filter, "
        f"{100 * n_flagged / n_measure:.1f}% of benchmark)"
    )
    consensus = {}
    for k in range(1, n_filters + 1):
        lo, hi = percentile_interval(resampled[k])
        items = observed.get(k, 0)
        consensus[k] = {
            "items": items,
            "pct": 100 * items / n_flagged,
            "ci": [100 * lo, 100 * hi],
        }
        if items or k <= 3:
            print(
                f"  exactly {k}/{n_filters}: {items:4d} items  "
                f"{consensus[k]['pct']:5.1f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]"
            )

    jaccard = {}
    firing = [name for name, flagged in flagged_by.items() if flagged]
    for left_name, right_name in itertools.combinations(firing, 2):
        left, right = flagged_by[left_name], flagged_by[right_name]
        jaccard[f"{left_name}|{right_name}"] = len(left & right) / len(left | right)
    print(f"  max pairwise Jaccard: {max(jaccard.values()):.3f}")

    # Intervals on the headline rates themselves. The control set has only 280
    # items, so the false-positive rates carry real uncertainty and should not
    # be quoted bare.
    rates = {}
    print(
        f"\nFlag rates with 95% CI (measurement n={n_measure}, control n={n_control})"
    )
    print(f"{'filter':22s} {'measurement':>22s} {'control (FPR)':>22s}")
    for name, flagged in main_results["filters"].items():
        rates[name] = {
            group: rate_with_interval(set(flagged[group]), total)
            for group, total in (("measure", n_measure), ("control", n_control))
        }
        measured, control_rate = rates[name]["measure"], rates[name]["control"]
        print(
            f"{name:22s} "
            f"{measured['pct']:6.1f}% [{measured['ci'][0]:4.1f},{measured['ci'][1]:5.1f}] "
            f"  {control_rate['pct']:6.1f}% "
            f"[{control_rate['ci'][0]:4.1f},{control_rate['ci'][1]:5.1f}]"
        )

    save(
        "consensus",
        {
            "flagged_any": n_flagged,
            "n": n_measure,
            "consensus": consensus,
            "jaccard": jaccard,
            "rates": rates,
        },
    )


def stage_scaling():
    """Figure 1 / Section 8: detection and error across corpus sizes."""
    measure_texts, control_texts = benchmark_texts()
    n_measure, n_control = len(measure_texts), len(control_texts)

    print(f"\nFigure 1  (measurement={n_measure}, control={n_control})")
    print(
        f"{'corpus':>8} | {'s1 det':>7} {'s1 fp':>6} {'frac':>5} | "
        f"{'ot3 det':>7} {'ot3 fp':>7} {'frac':>5} | {'fuzz':>4}"
    )

    rows = []
    for size in SCALES:
        docs = corpus(size)
        s1_measure = len(flag_by_ngram(docs, measure_texts, 8, tok_s1))
        s1_control = len(flag_by_ngram(docs, control_texts, 8, tok_s1))
        ot3_measure = len(flag_by_ngram(docs, measure_texts, 13, tok_qwen))
        ot3_control = len(flag_by_ngram(docs, control_texts, 13, tok_qwen))
        fuzz_hits = len(flag_by_fuzz(docs, measure_texts, 95.0)) + len(
            flag_by_fuzz(docs, control_texts, 95.0)
        )

        # The error fraction is the false-positive rate over the detection rate:
        # what share of what the filter flags is provably wrong.
        row = {
            "corpus": size,
            "s1_det": 100 * s1_measure / n_measure,
            "s1_fp": 100 * s1_control / n_control,
            "s1_frac": (
                (s1_control / n_control) / (s1_measure / n_measure)
                if s1_measure
                else None
            ),
            "ot3_det": 100 * ot3_measure / n_measure,
            "ot3_fp": 100 * ot3_control / n_control,
            "ot3_frac": (
                (ot3_control / n_control) / (ot3_measure / n_measure)
                if ot3_measure
                else None
            ),
            "fuzz": fuzz_hits,
        }
        rows.append(row)
        print(
            f"{size:8d} | {row['s1_det']:6.1f}% {row['s1_fp']:5.1f}% "
            f"{row['s1_frac']:5.2f} | {row['ot3_det']:6.1f}% {row['ot3_fp']:6.1f}% "
            f"{row['ot3_frac']:5.2f} | {fuzz_hits:4d}"
        )
    save("scaling", rows)


def stage_tuning():
    """Figure 2 / Section 10: parameter sweeps against the control."""
    docs = corpus(MAIN_N)
    measure_texts, control_texts = benchmark_texts()
    n_measure, n_control = len(measure_texts), len(control_texts)
    sweeps = {}

    for label, tokenize in [("word", tok_s1), ("qwen", tok_qwen)]:
        # Index every n at once, then make a single tokenizing pass over the
        # corpus. Calling flag_by_ngram per n would re-tokenize it 16 times.
        index = {}
        for group, texts in (("measure", measure_texts), ("control", control_texts)):
            tokenized = [tokenize(t) for t in texts]
            for n in NGRAM_SIZES:
                by_gram = {}
                for item, tokens in enumerate(tokenized):
                    for gram in ngrams(tokens, n):
                        by_gram.setdefault(gram, set()).add(item)
                index[(group, n)] = by_gram

        GROUPS = ("measure", "control")
        flagged = {(group, n): set() for group in GROUPS for n in NGRAM_SIZES}
        for text in docs:
            if not text.strip():
                continue
            tokens = tokenize(text)
            for n in NGRAM_SIZES:
                grams = ngrams(tokens, n)
                if not grams:
                    continue
                for group in GROUPS:
                    by_gram = index[(group, n)]
                    for gram in grams:
                        hit = by_gram.get(gram)
                        if hit:
                            flagged[(group, n)] |= hit

        rows = []
        print(f"\nFigure 2, {label} n-gram sweep")
        print(f"{'n':>4} | {'detect':>7} {'fp':>7} {'frac':>6}")
        for n in NGRAM_SIZES:
            detect = len(flagged[("measure", n)]) / n_measure
            fp = len(flagged[("control", n)]) / n_control
            rows.append(
                {
                    "n": n,
                    "detect": 100 * detect,
                    "fp": 100 * fp,
                    "frac": (fp / detect) if detect else None,
                }
            )
            ratio = f"{fp / detect:6.2f}" if detect else "     -"
            print(f"{n:4d} | {100 * detect:6.1f}% {100 * fp:6.1f}% {ratio}")
        sweeps[label] = rows

    # The fuzzy filter scores whole strings, so one score matrix per group
    # answers every threshold at once.
    from rapidfuzz import fuzz, process

    best_measure = process.cdist(
        docs, measure_texts, scorer=fuzz.ratio, score_cutoff=50, workers=-1
    ).max(axis=0)
    best_control = process.cdist(
        docs, control_texts, scorer=fuzz.ratio, score_cutoff=50, workers=-1
    ).max(axis=0)
    rows = []
    print("\nFigure 2, fuzz.ratio sweep")
    print(f"{'thr':>4} | {'detect':>7} {'fp':>7} {'frac':>6}")
    for threshold in FUZZ_THRESHOLDS:
        detect = (best_measure >= threshold).sum() / n_measure
        fp = (best_control >= threshold).sum() / n_control
        rows.append(
            {
                "threshold": threshold,
                "detect": 100 * detect,
                "fp": 100 * fp,
                "frac": (fp / detect) if detect else None,
            }
        )
        ratio = f"{fp / detect:6.2f}" if detect else "     -"
        print(f"{threshold:4d} | {100 * detect:6.1f}% {100 * fp:6.1f}% {ratio}")
    sweeps["fuzz"] = rows
    save("tuning", sweeps)


def stage_mechanism():
    """Section 9: which n-grams cause the false positives."""
    corpus_texts = corpus(MAIN_N)
    _, control_texts = benchmark_texts()

    by_gram = {}
    for item, text in enumerate(control_texts):
        for gram in ngrams(tok_s1(text), 8):
            by_gram.setdefault(gram, set()).add(item)

    # For each 8-gram shared with the control set: how many corpus documents
    # contain it, and which control items it therefore flags.
    doc_count = collections.Counter()
    items_hit = collections.defaultdict(set)
    for text in corpus_texts:
        if not text.strip():
            continue
        for gram in ngrams(tok_s1(text), 8):
            hit = by_gram.get(gram)
            if hit:
                doc_count[gram] += 1
                items_hit[gram] |= hit

    false_positives = set().union(*items_hit.values()) if items_hit else set()
    print(
        f"\nSection 9: {len(doc_count)} distinct 8-grams cause "
        f"{len(false_positives)}/{len(control_texts)} false positives"
    )

    # most_common breaks ties by insertion order, and these n-grams were
    # inserted while iterating a set, so equal counts came out in hash order and
    # the file was not byte-identical between runs. Break ties on the phrase.
    ranked = sorted(doc_count.items(), key=lambda kv: (-kv[1], kv[0]))[:15]

    top = []
    for gram, docs in ranked:
        phrase = " ".join(gram)
        top.append({"ngram": phrase, "docs": docs, "items": len(items_hit[gram])})
        print(f'  {docs:5d} docs, {len(items_hit[gram]):2d} items  "{phrase}"')

    save(
        "mechanism",
        {
            "distinct_ngrams": len(doc_count),
            "false_positives": len(false_positives),
            "n_control": len(control_texts),
            "top": top,
        },
    )


def stage_composition():
    """Section 11: which Dolci slice each flagged item came from."""
    corpus_rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAIN_N:
                break
            corpus_rows.append(json.loads(line))
    measure, _ = benchmarks()
    measure_texts = [r["problem"] for r in measure]
    slice_sizes = collections.Counter(r["source"] for r in corpus_rows)

    by_gram = {}
    for item, text in enumerate(measure_texts):
        for gram in ngrams(tok_qwen(text), 13):
            by_gram.setdefault(gram, set()).add(item)

    # Attribute each flag to the Dolci slice the matching document came from.
    docs_per_slice = collections.Counter()
    items_per_slice = collections.defaultdict(set)
    for row in corpus_rows:
        text = row["problem"]
        if not text.strip():
            continue
        hit = set()
        for gram in ngrams(tok_qwen(text), 13):
            matched = by_gram.get(gram)
            if matched:
                hit |= matched
        if hit:
            docs_per_slice[row["source"]] += 1
            items_per_slice[row["source"]] |= hit

    n_flagged = len(set().union(*items_per_slice.values())) if items_per_slice else 0
    print(f"\nSection 11 (OpenThoughts3 filter, {n_flagged} items flagged)")
    print(f"{'slice':40s} {'corpus%':>8} {'docs':>6} {'items':>6} {'enrich':>7}")

    slices = []
    for source, n_docs in docs_per_slice.most_common():
        # Enrichment: this slice's share of the flags over its share of the
        # corpus. Above 1 means the slice flags more than its size explains.
        corpus_share = slice_sizes[source] / len(corpus_rows)
        flag_share = len(items_per_slice[source]) / n_flagged if n_flagged else 0
        slices.append(
            {
                "source": source,
                "corpus_share": 100 * corpus_share,
                "docs": n_docs,
                "items": len(items_per_slice[source]),
                "enrichment": (flag_share / corpus_share if corpus_share else None),
            }
        )
        print(
            f"  {source[:38]:38s} {100 * corpus_share:7.1f}% {n_docs:6d} "
            f"{len(items_per_slice[source]):6d} {flag_share / corpus_share:6.1f}x"
        )
    save("composition", {"flagged": n_flagged, "slices": slices})


def stage_validate():
    """Test the assumption the control set rests on.

    The method assumes a benchmark published after the corpus was assembled cannot
    be in it, so the corpus release date is an upper bound on its contents. Dolci
    is a remix of separately curated corpora, so in principle a component could
    postdate the release of the whole. If any control item is genuinely present,
    its flags are not false positives and the method breaks.

    A genuinely present item shares long spans with some training document. A
    coincidental one shares only stock phrasing. So: for each benchmark item find
    the longest n for which it shares an n-gram with any corpus document, and
    compare the two groups. Any control item matching at long n is inspected.
    """
    docs = corpus(MAIN_N)
    measure, control = benchmarks()

    corpus_tokens = [tok_s1(t) for t in docs if t.strip()]
    distributions, longest = {}, {}
    for label, records in (("measurement", measure), ("control", control)):
        texts = [r["problem"] for r in records]
        index = {n: {} for n in SPAN_SIZES}
        for n in SPAN_SIZES:
            for item, text in enumerate(texts):
                for gram in ngrams(tok_s1(text), n):
                    index[n].setdefault(gram, set()).add(item)

        # An item usually has several spans tied at its longest n. `ngrams`
        # returns a set, which iterates in hash order and so varies between
        # processes; take the lexicographically smallest span at each n to keep
        # the reported example stable across runs.
        best_n, best_span = {}, {}
        for tokens in corpus_tokens:
            for n in SPAN_SIZES:
                for gram in ngrams(tokens, n):
                    for item in index[n].get(gram, ()):
                        candidate = " ".join(gram)
                        previous = best_n.get(item)
                        if (
                            previous is None
                            or n > previous
                            or (n == previous and candidate < best_span[item])
                        ):
                            best_n[item] = n
                            best_span[item] = candidate

        distribution = collections.Counter(
            best_n.get(item, 0) for item in range(len(texts))
        )
        distributions[label] = {
            str(n): count for n, count in sorted(distribution.items())
        }

        top = sorted(best_n, key=lambda item: (-best_n[item], item))[:5]
        longest[label] = [
            {
                "item": records[item]["id"],
                "benchmark": records[item].get("benchmark", "olymmath"),
                "n": best_n[item],
                "span": best_span[item],
            }
            for item in top
        ]

        print(f"\n{label} ({len(texts)} items): longest shared word n-gram with corpus")
        for n in sorted(distribution):
            label_n = "no match" if n == 0 else f"n >= {n}"
            print(f"  {label_n:>10}: {distribution[n]:4d} items")

    print("\nLongest control spans -- must be stock phrasing, not problem content:")
    for entry in longest["control"]:
        print(f'  n={entry["n"]:2d}  {entry["benchmark"]:16s} "{entry["span"][:86]}"')

    save("validate", {"distribution": distributions, "longest": longest})


def stage_second():
    """Replicate the false-positive measurement on a second, older corpus.

    The Tulu 3 SFT mixture shipped November 2024, so the date split moves with it:
    every 2025 competition becomes an additional control, not just the 2026 ones.
    That leaves only the 2024 competitions as a measurement set, which is small,
    but the quantity of interest here is the false-positive rate, and for that the
    control side is what matters -- and it is much larger than for Dolci.

    If the error rate is a property of the method rather than of Dolci, it should
    reappear at a similar level here.
    """
    docs = []
    with open("data/raw/tulu_math.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAIN_N:
                break
            docs.append(json.loads(line)["problem"])

    all_measure, control_2026 = benchmarks()
    # Competitions held in 2024 are the only ones that predate this corpus.
    from_2024 = [
        r
        for r in all_measure
        if r.get("benchmark", "") in ("aime_2024_I", "aime_2024_II", "hmmt_feb_2024")
    ]
    # Everything from 2025 postdates it, so it joins the control side.
    from_2025_on = [r for r in all_measure if r not in from_2024] + control_2026

    measure_texts = [r["problem"] for r in from_2024]
    control_texts = [r["problem"] for r in from_2025_on]
    n_measure, n_control = len(measure_texts), len(control_texts)

    print(f"\nSecond corpus: Tulu 3 SFT math ({len(docs)} docs, released Nov 2024)")
    print(f"  measurement (2024 competitions): {n_measure} items")
    print(f"  control (2025 and 2026, all postdating the corpus): {n_control} items")
    print(f"\n{'filter':24s} {'detection':>12s} {'false positive':>16s}")

    results = {}
    for name, _provenance, run in filters():
        flagged_measure = run(docs, measure_texts)
        flagged_control = run(docs, control_texts)
        results[name] = {
            "detect": 100 * len(flagged_measure) / n_measure,
            "fp": 100 * len(flagged_control) / n_control,
            "n_detect": len(flagged_measure),
            "n_fp": len(flagged_control),
        }
        print(
            f"{name:24s} "
            f"{len(flagged_measure):4d} ({100 * len(flagged_measure) / n_measure:5.1f}%) "
            f"{len(flagged_control):6d} ({100 * len(flagged_control) / n_control:5.1f}%)"
        )

    save(
        "second_corpus",
        {
            "corpus": "tulu_math",
            "docs": len(docs),
            "n_measure": n_measure,
            "n_control": n_control,
            "filters": results,
        },
    )


def stage_matched():
    """Matched same-competition comparison (Sec. matched).

    The estimated false-discovery share assumes the measurement and control groups
    attract spurious matches at comparable rates. That is testable directly: AIME
    2025 and AIME 2026 are the same competition one year apart, written to the same
    conventions, and only one of them can be in the corpus. If both flag at similar
    rates, most of what is flagged on the measurement side is boilerplate too.
    """
    docs = corpus(MAIN_N)
    pairs = [("aime_2025", "aime_2026"), ("hmmt_feb_2025", "hmmt_feb_2026")]
    by_competition = {}
    for record in load_jsonl("data/raw/bench_measure.jsonl") + load_jsonl(
        "data/raw/bench_control.jsonl"
    ):
        by_competition.setdefault(record["benchmark"], []).append(record["problem"])

    results = {}
    print("\nMatched same-competition comparison (100k Dolci documents)")
    print(f"{'competition':18s} {'n':>4} {'s1':>8} {'OT3':>8}")
    for older, newer in pairs:
        for competition in (older, newer):
            texts = by_competition.get(competition, [])
            if not texts:
                print(f"  {competition}: NOT FOUND")
                continue
            s1 = len(flag_by_ngram(docs, texts, 8, tok_s1)) / len(texts)
            ot3 = len(flag_by_ngram(docs, texts, 13, tok_qwen)) / len(texts)
            side = "measure" if competition == older else "control"
            results[competition] = {
                "n": len(texts),
                "s1": 100 * s1,
                "ot3": 100 * ot3,
                "side": side,
            }
            print(
                f"  {competition:16s} {len(texts):4d} "
                f"{100 * s1:7.1f}% {100 * ot3:7.1f}%   ({side})"
            )
    save("matched", results)


def stage_nemotron():
    """Third corpus: Nemotron-SFT-Math-v4, released 2026-05-01.

    Its release date sits between the two groups used for Dolci, so the split is
    recomputed rather than reused. Everything published before May 2026 could be
    inside it and becomes measurement; only June 2026 onward can serve as control.
    """
    docs = []
    with open("data/raw/nemotron_questions.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAIN_N:
                break
            docs.append(json.loads(line)["problem"])

    measure_dolci, control_dolci = benchmarks()
    early_2026 = [r for r in control_dolci if r.get("benchmark") in PRE_MAY_2026]
    control = load_jsonl("data/raw/bench_control_post_may26.jsonl")

    measure_texts = [r["problem"] for r in measure_dolci + early_2026]
    control_texts = [r["problem"] for r in control]
    n_measure, n_control = len(measure_texts), len(control_texts)

    print(
        f"\nThird corpus: Nemotron-SFT-Math-v4 ({len(docs)} docs, released 2026-05-01)"
    )
    print(f"  measurement (published before May 2026): {n_measure} items")
    print(
        f"    = {len(measure_dolci)} from the Dolci measurement set "
        f"+ {len(early_2026)} early-2026 sets that postdate Dolci but not Nemotron"
    )
    print(f"  control (June 2026 onward): {n_control} items")
    print(f"\n{'filter':24s} {'detection':>12s} {'false positive':>16s}")

    results = {}
    for name, _provenance, run in filters():
        flagged_measure = run(docs, measure_texts)
        flagged_control = run(docs, control_texts)
        detect = 100 * len(flagged_measure) / n_measure
        fp = 100 * len(flagged_control) / n_control
        results[name] = {
            "detect": detect,
            "fp": fp,
            "n_detect": len(flagged_measure),
            "n_fp": len(flagged_control),
            "fd_share": (fp / detect) if detect else None,
        }
        print(
            f"{name:24s} {len(flagged_measure):5d} ({detect:5.1f}%) "
            f"{len(flagged_control):7d} ({fp:5.1f}%)"
        )
    save(
        "nemotron",
        {
            "corpus": "Nemotron-SFT-Math-v4",
            "docs": len(docs),
            "n_measure": n_measure,
            "n_control": n_control,
            "filters": results,
        },
    )


def stage_phrases():
    """Table 5 and the phrase-agreement series in Fig. 2.

    Sec. 9 predicts that requiring agreement across spans spread through an item
    should suppress the single-stock-phrase match. This sweeps the fraction of an
    item's twenty sampled phrases that must be present, which is the parameter
    that governs how much agreement is demanded.
    """
    docs = corpus(MAIN_N)
    measure_texts, control_texts = benchmark_texts()
    n_measure, n_control = len(measure_texts), len(control_texts)

    print()
    print(
        "Phrase agreement: 20 sampled 8-grams per item, sweeping the "
        f"required fraction ({len(docs)} documents)"
    )
    print(f"{'required':>9} {'detect':>9} {'fp':>9} {'fd share':>9}")

    rows = []
    for required in PHRASE_FRACTIONS:
        flagged_measure = flag_by_sampled_phrases(docs, measure_texts, frac=required)
        flagged_control = flag_by_sampled_phrases(docs, control_texts, frac=required)
        detect = 100 * len(flagged_measure) / n_measure
        fp = 100 * len(flagged_control) / n_control
        rows.append(
            {
                "required": required,
                "detect": detect,
                "fp": fp,
                "frac": (fp / detect) if detect else None,
            }
        )
        share = f"{fp / detect:9.2f}" if detect else f"{'--':>9}"
        print(f"{required:9.2f} {detect:8.1f}% {fp:8.1f}% {share}")
    save("phrases", rows)


def stage_strata():
    """The control set's two genres, scored separately.

    The measurement set is almost entirely competition problems, but only 69 of
    the 280 controls are. Pooling the two genres into one false-positive rate
    averages populations the filters treat very differently, and the weighting
    is an accident of which MathArena sets exist for 2026.
    """
    COMPETITION = {"aime_2026", "hmmt_feb_2026", "usamo_2026"}

    _, control = benchmarks()
    main = json.loads((OUT / "main.json").read_text(encoding="utf-8"))

    genre = [
        "competition" if r.get("benchmark") in COMPETITION else "research"
        for r in control
    ]
    sizes = collections.Counter(genre)

    rows = {}
    print(f"\nControl set by genre (competition={sizes['competition']}, "
          f"research={sizes['research']})")
    print(f"{'filter':26s} {'competition':>18s} {'research':>18s}")
    for name, flagged in main["filters"].items():
        counts = collections.Counter(genre[i] for i in flagged["control"])
        rows[name] = {
            stratum: {
                "n": sizes[stratum],
                "flagged": counts[stratum],
                "pct": 100 * counts[stratum] / sizes[stratum] if sizes[stratum] else 0.0,
            }
            for stratum in ("competition", "research")
        }
        comp, res = rows[name]["competition"], rows[name]["research"]
        print(
            f"{name:26s} {comp['flagged']:5d} ({comp['pct']:5.1f}%) "
            f"  {res['flagged']:5d} ({res['pct']:5.1f}%)"
        )

    out = {"sizes": dict(sizes), "filters": rows}
    (OUT / "strata.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    _log.info("wrote %s", OUT / "strata.json")


# ----------------------------------------------------------- entry point

# Ordered as the paper presents them; `--stage` runs one, no argument runs all.

STAGES = {
    "main": stage_main,
    "strata": stage_strata,
    "consensus": stage_consensus,
    "scaling": stage_scaling,
    "tuning": stage_tuning,
    "mechanism": stage_mechanism,
    "composition": stage_composition,
    "validate": stage_validate,
    "second": stage_second,
    "matched": stage_matched,
    "nemotron": stage_nemotron,
    "phrases": stage_phrases,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", choices=list(STAGES), help="run a single stage instead of all twelve"
    )
    args = parser.parse_args()
    configure_logging()

    for name in [args.stage] if args.stage else STAGES:
        STAGES[name]()


if __name__ == "__main__":
    main()

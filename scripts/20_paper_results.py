"""Regenerate every number reported in paper/arr_decontamination.tex.

Each stage writes a JSON file under results/paper/ and prints the figures as the
paper states them, so a reader can diff the two. Stages are independent; run one
with --stage or all of them with no argument.

  python scripts/20_paper_results.py --stage main         # Table 2
  python scripts/20_paper_results.py --stage consensus    # Sec. 6 agreement
  python scripts/20_paper_results.py --stage scaling      # Fig. 1, Sec. 8
  python scripts/20_paper_results.py --stage tuning       # Fig. 2, Sec. 10
  python scripts/20_paper_results.py --stage mechanism    # Sec. 9
  python scripts/20_paper_results.py --stage composition  # Sec. 11

decon is not run here: it is a separate Rust binary. See scripts/21_run_decon.py.
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
    flag_by_coverage, flag_by_fuzz, flag_by_ngram, ngrams,
    tok_s1, tok_whitespace, _hf_tokenizer,
)
from contamination_audit.io import load_jsonl

_log = logging.getLogger("paper_results")

OUT = Path("results/paper")
CORPUS = "data/raw/dolci_questions.jsonl"
MAIN_N = 100_000
SCALES = [12_500, 25_000, 50_000, 100_000, 150_000, 250_000, 390_000]
SEED = 42
BOOTSTRAP = 10_000


# ----------------------------------------------------------------- loading

def corpus(n: int) -> list[str]:
    out = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(json.loads(line)["problem"])
    return out


def benchmarks() -> tuple[list[dict], list[dict]]:
    olym = [
        {"id": r["id"], "benchmark": "olymmath", "problem": r["problem"]}
        for r in load_jsonl("data/raw/olymmath_en_hard.jsonl")
        + load_jsonl("data/raw/olymmath_en_easy.jsonl")
    ]
    measure = olym + load_jsonl("data/raw/bench_measure.jsonl")
    control = load_jsonl("data/raw/bench_control.jsonl")
    return measure, control


def filters(qwen):
    return [
        ("s1-8gram-word", "code", lambda c, b: flag_by_ngram(c, b, 8, tok_s1)),
        ("ot3-13gram-qwen", "code", lambda c, b: flag_by_ngram(c, b, 13, qwen)),
        ("ot114k-fuzz95", "code", lambda c, b: flag_by_fuzz(c, b, 95.0)),
        ("tulu3-8gram-50pct", "prose",
         lambda c, b: flag_by_coverage(c, b, 8, 0.5, tok_whitespace)),
    ]


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    _log.info("wrote %s", OUT / f"{name}.json")


# ----------------------------------------------------------------- stages

def stage_main(qwen):
    """Table 2: flag counts per filter on both benchmark groups."""
    c = corpus(MAIN_N)
    measure, control = benchmarks()
    mt = [r["problem"] for r in measure]
    ct = [r["problem"] for r in control]
    res = {}
    print(f"\nTable 2  (corpus={len(c)}, measurement={len(mt)}, control={len(ct)})")
    print(f"{'filter':24s} {'params':6s} {'measurement':>14s} {'control':>14s}")
    for name, prov, fn in filters(qwen):
        m, k = fn(c, mt), fn(c, ct)
        res[name] = {"provenance": prov, "measure": sorted(m), "control": sorted(k)}
        print(f"{name:24s} {prov:6s} {len(m):5d} ({100*len(m)/len(mt):5.1f}%) "
              f"{len(k):5d} ({100*len(k)/len(ct):5.1f}%)")
    print(f"{'decon (see 21_run_decon)':24s} {'code':6s} {0:5d} (  0.0%) {0:5d} (  0.0%)")
    save("main", {"corpus_n": len(c), "n_measure": len(mt), "n_control": len(ct),
                  "filters": res})


def stage_consensus():
    """Section 6: how many filters agree on each flagged item."""
    d = json.load(open(OUT / "main.json", encoding="utf-8"))
    N = d["n_measure"]
    flags = {k: set(v["measure"]) for k, v in d["filters"].items()}
    flags["decon"] = set()
    K = len(flags)

    def tally(idxs):
        v = collections.Counter()
        for i in idxs:
            k = sum(i in s for s in flags.values())
            if k:
                v[k] += 1
        return v, sum(v.values())

    v, tot = tally(range(N))
    rng = random.Random(SEED)
    boot = {k: [] for k in range(1, K + 1)}
    for _ in range(BOOTSTRAP):
        idxs = [rng.randrange(N) for _ in range(N)]
        vv, tt = tally(idxs)
        for k in range(1, K + 1):
            boot[k].append(vv.get(k, 0) / tt if tt else 0.0)

    def ci(x):
        x = sorted(x)
        return x[int(.025 * len(x))], x[min(int(.975 * len(x)), len(x) - 1)]

    print(f"\nSection 6  ({tot}/{N} items flagged by >=1 filter, "
          f"{100*tot/N:.1f}% of benchmark)")
    rows = {}
    for k in range(1, K + 1):
        lo, hi = ci(boot[k])
        rows[k] = {"items": v.get(k, 0), "pct": 100 * v.get(k, 0) / tot,
                   "ci": [100 * lo, 100 * hi]}
        if v.get(k) or k <= 3:
            print(f"  exactly {k}/{K}: {v.get(k,0):4d} items  {rows[k]['pct']:5.1f}%"
                  f"  95% CI [{100*lo:.1f}, {100*hi:.1f}]")
    pairs = {}
    names = [k for k in flags if flags[k]]
    for a, b in itertools.combinations(names, 2):
        A, B = flags[a], flags[b]
        pairs[f"{a}|{b}"] = len(A & B) / len(A | B)
    print("  max pairwise Jaccard: %.3f" % max(pairs.values()))

    # Bootstrap intervals on the headline rates themselves. The control set has
    # only 280 items, so the false-positive rates carry real uncertainty and
    # should not be quoted bare.
    NC = d["n_control"]
    rates = {}
    print(f"\nFlag rates with 95% CI (measurement n={N}, control n={NC})")
    print(f"{'filter':22s} {'measurement':>22s} {'control (FPR)':>22s}")
    for name, v in d["filters"].items():
        row = {}
        for grp, total in (("measure", N), ("control", NC)):
            flagged = set(v[grp])
            # Resample the observed indicator vector rather than simulating a
            # binomial: nonparametric, and it makes no distributional assumption.
            ind = [1 if i in flagged else 0 for i in range(total)]
            r = random.Random(SEED)
            boot = sorted(100 * sum(r.choices(ind, k=total)) / total
                          for _ in range(BOOTSTRAP))
            k = len(flagged)
            row[grp] = {"pct": 100 * k / total,
                        "ci": [boot[int(.025 * BOOTSTRAP)],
                               boot[int(.975 * BOOTSTRAP)]]}
        rates[name] = row
        m, cc = row["measure"], row["control"]
        print(f"{name:22s} {m['pct']:6.1f}% [{m['ci'][0]:4.1f},{m['ci'][1]:5.1f}] "
              f"  {cc['pct']:6.1f}% [{cc['ci'][0]:4.1f},{cc['ci'][1]:5.1f}]")

    save("consensus", {"flagged_any": tot, "n": N, "consensus": rows,
                       "jaccard": pairs, "rates": rates})


def stage_scaling(qwen):
    """Figure 1 / Section 8: detection and error across corpus sizes."""
    measure, control = benchmarks()
    mt = [r["problem"] for r in measure]
    ct = [r["problem"] for r in control]
    M, C = len(mt), len(ct)
    rows = []
    print(f"\nFigure 1  (measurement={M}, control={C})")
    print(f"{'corpus':>8} | {'s1 det':>7} {'s1 fp':>6} {'frac':>5} | "
          f"{'ot3 det':>7} {'ot3 fp':>7} {'frac':>5} | {'fuzz':>4}")
    for n in SCALES:
        c = corpus(n)
        s1m = len(flag_by_ngram(c, mt, 8, tok_s1))
        s1c = len(flag_by_ngram(c, ct, 8, tok_s1))
        o3m = len(flag_by_ngram(c, mt, 13, qwen))
        o3c = len(flag_by_ngram(c, ct, 13, qwen))
        fz = len(flag_by_fuzz(c, mt, 95.0)) + len(flag_by_fuzz(c, ct, 95.0))
        r = {"corpus": n,
             "s1_det": 100*s1m/M, "s1_fp": 100*s1c/C, "s1_frac": (s1c/C)/(s1m/M) if s1m else None,
             "ot3_det": 100*o3m/M, "ot3_fp": 100*o3c/C, "ot3_frac": (o3c/C)/(o3m/M) if o3m else None,
             "fuzz": fz}
        rows.append(r)
        print(f"{n:8d} | {r['s1_det']:6.1f}% {r['s1_fp']:5.1f}% {r['s1_frac']:5.2f} | "
              f"{r['ot3_det']:6.1f}% {r['ot3_fp']:6.1f}% {r['ot3_frac']:5.2f} | {fz:4d}")
    save("scaling", rows)


def stage_tuning(qwen):
    """Figure 2 / Section 10: parameter sweeps against the control."""
    c = corpus(MAIN_N)
    measure, control = benchmarks()
    mt = [r["problem"] for r in measure]
    ct = [r["problem"] for r in control]
    M, C = len(mt), len(ct)
    out = {}

    NS = [5, 8, 10, 13, 15, 20, 25, 30]
    for label, tokenize in [("word", tok_s1), ("qwen", qwen)]:
        # Tokenize each document once and test every n against that one pass.
        # Calling flag_by_ngram per n would re-tokenize the corpus 16 times.
        lk = {}
        for grp, texts in (("m", mt), ("c", ct)):
            toks = [tokenize(t) for t in texts]
            for n in NS:
                d = {}
                for j, tk in enumerate(toks):
                    for g in ngrams(tk, n):
                        d.setdefault(g, set()).add(j)
                lk[(grp, n)] = d

        hit = {(g, n): set() for g in "mc" for n in NS}
        for txt in c:
            if not txt.strip():
                continue
            tk = tokenize(txt)
            for n in NS:
                gs = ngrams(tk, n)
                if not gs:
                    continue
                for grp in "mc":
                    d = lk[(grp, n)]
                    for g in gs:
                        h = d.get(g)
                        if h:
                            hit[(grp, n)] |= h

        rows = []
        print(f"\nFigure 2, {label} n-gram sweep")
        print(f"{'n':>4} | {'detect':>7} {'fp':>7} {'frac':>6}")
        for n in NS:
            dm = len(hit[("m", n)]) / M
            fp = len(hit[("c", n)]) / C
            rows.append({"n": n, "detect": 100*dm, "fp": 100*fp,
                         "frac": (fp/dm) if dm else None})
            frac = f"{fp/dm:6.2f}" if dm else "     -"
            print(f"{n:4d} | {100*dm:6.1f}% {100*fp:6.1f}% {frac}")
        out[label] = rows

    from rapidfuzz import fuzz, process
    mm = process.cdist(c, mt, scorer=fuzz.ratio, score_cutoff=50, workers=-1).max(axis=0)
    mc = process.cdist(c, ct, scorer=fuzz.ratio, score_cutoff=50, workers=-1).max(axis=0)
    rows = []
    print("\nFigure 2, fuzz.ratio sweep")
    print(f"{'thr':>4} | {'detect':>7} {'fp':>7} {'frac':>6}")
    for thr in [95, 90, 85, 80, 75, 70, 65, 60]:
        dm = (mm >= thr).sum() / M
        fp = (mc >= thr).sum() / C
        rows.append({"threshold": thr, "detect": 100*dm, "fp": 100*fp,
                     "frac": (fp/dm) if dm else None})
        frac = f"{fp/dm:6.2f}" if dm else "     -"
        print(f"{thr:4d} | {100*dm:6.1f}% {100*fp:6.1f}% {frac}")
    out["fuzz"] = rows
    save("tuning", out)


def stage_mechanism():
    """Section 9: which n-grams cause the false positives."""
    c = corpus(MAIN_N)
    _, control = benchmarks()
    ct = [r["problem"] for r in control]
    lk = {}
    for j, t in enumerate(ct):
        for g in ngrams(tok_s1(t), 8):
            lk.setdefault(g, set()).add(j)
    docs = collections.Counter()
    items = collections.defaultdict(set)
    for txt in c:
        if not txt.strip():
            continue
        for g in ngrams(tok_s1(txt), 8):
            h = lk.get(g)
            if h:
                docs[g] += 1
                items[g] |= h
    hit = set().union(*items.values()) if items else set()
    print(f"\nSection 9: {len(docs)} distinct 8-grams cause "
          f"{len(hit)}/{len(ct)} false positives")
    top = []
    for g, n in docs.most_common(15):
        top.append({"ngram": " ".join(g), "docs": n, "items": len(items[g])})
        print(f"  {n:5d} docs, {len(items[g]):2d} items  \"{' '.join(g)}\"")
    save("mechanism", {"distinct_ngrams": len(docs), "false_positives": len(hit),
                       "n_control": len(ct), "top": top})


def stage_second(qwen):
    """Replicate the false-positive measurement on a second, older corpus.

    The Tulu 3 SFT mixture shipped November 2024, so the date split moves with it:
    every 2025 competition becomes an additional control, not just the 2026 ones.
    That leaves only the 2024 competitions as a measurement set, which is small,
    but the quantity of interest here is the false-positive rate, and for that the
    control side is what matters -- and it is much larger than for Dolci.

    If the error rate is a property of the method rather than of Dolci, it should
    reappear at a similar level here.
    """
    rows = []
    with open("data/raw/tulu_math.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAIN_N:
                break
            rows.append(json.loads(line)["problem"])

    measure_all, control_2026 = benchmarks()
    # Competitions held in 2024 are the only ones that predate this corpus.
    pre2024 = [r for r in measure_all
               if r.get("benchmark", "") in ("aime_2024_I", "aime_2024_II", "hmmt_feb_2024")]
    # Everything from 2025 postdates it, so it joins the control side.
    post = [r for r in measure_all if r not in pre2024] + control_2026

    mt = [r["problem"] for r in pre2024]
    ct = [r["problem"] for r in post]
    print(f"\nSecond corpus: Tulu 3 SFT math ({len(rows)} docs, released Nov 2024)")
    print(f"  measurement (2024 competitions): {len(mt)} items")
    print(f"  control (2025 and 2026, all postdating the corpus): {len(ct)} items")
    print(f"\n{'filter':24s} {'detection':>12s} {'false positive':>16s}")
    out = {}
    for name, prov, fn in filters(qwen):
        m, k = fn(rows, mt), fn(rows, ct)
        out[name] = {"detect": 100*len(m)/len(mt), "fp": 100*len(k)/len(ct),
                     "n_detect": len(m), "n_fp": len(k)}
        print(f"{name:24s} {len(m):4d} ({100*len(m)/len(mt):5.1f}%) "
              f"{k and len(k) or 0:6d} ({100*len(k)/len(ct):5.1f}%)")
    save("second_corpus", {"corpus": "tulu_math", "docs": len(rows),
                           "n_measure": len(mt), "n_control": len(ct),
                           "filters": out})


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
    c = corpus(MAIN_N)
    measure, control = benchmarks()
    NS = [8, 10, 13, 15, 20, 25, 30, 40]

    corpus_toks = [tok_s1(t) for t in c if t.strip()]
    out, detail = {}, {}
    for label, rows in (("measurement", measure), ("control", control)):
        texts = [r["problem"] for r in rows]
        idx = {n: {} for n in NS}
        for n in NS:
            for j, t in enumerate(texts):
                for g in ngrams(tok_s1(t), n):
                    idx[n].setdefault(g, set()).add(j)

        best, span = {}, {}
        for tk in corpus_toks:
            for n in NS:
                for g in ngrams(tk, n):
                    for j in idx[n].get(g, ()):
                        if n > best.get(j, 0):
                            best[j] = n
                            span[j] = " ".join(g)

        dist = collections.Counter(best.get(j, 0) for j in range(len(texts)))
        out[label] = {str(k): v for k, v in sorted(dist.items())}
        top = sorted(best, key=lambda j: -best[j])[:5]
        detail[label] = [{"item": rows[j]["id"],
                          "benchmark": rows[j].get("benchmark", "olymmath"),
                          "n": best[j], "span": span[j]} for j in top]

        print(f"\n{label} ({len(texts)} items): longest shared word n-gram with corpus")
        for k in sorted(dist):
            lab = "no match" if k == 0 else f"n >= {k}"
            print(f"  {lab:>10}: {dist[k]:4d} items")

    print("\nLongest control spans -- must be stock phrasing, not problem content:")
    for d in detail["control"]:
        print(f"  n={d['n']:2d}  {d['benchmark']:16s} \"{d['span'][:86]}\"")

    save("validate", {"distribution": out, "longest": detail})



def stage_composition(qwen):
    """Section 11: which Dolci slice each flagged item came from."""
    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAIN_N:
                break
            rows.append(json.loads(line))
    measure, _ = benchmarks()
    mt = [r["problem"] for r in measure]
    totals = collections.Counter(r["source"] for r in rows)

    lk = {}
    for j, t in enumerate(mt):
        for g in ngrams(qwen(t), 13):
            lk.setdefault(g, set()).add(j)
    docs = collections.Counter()
    items = collections.defaultdict(set)
    for r in rows:
        txt = r["problem"]
        if not txt.strip():
            continue
        hit = set()
        for g in ngrams(qwen(txt), 13):
            h = lk.get(g)
            if h:
                hit |= h
        if hit:
            docs[r["source"]] += 1
            items[r["source"]] |= hit
    flagged = len(set().union(*items.values())) if items else 0
    print(f"\nSection 11 (OpenThoughts3 filter, {flagged} items flagged)")
    print(f"{'slice':40s} {'corpus%':>8} {'docs':>6} {'items':>6} {'enrich':>7}")
    out = []
    for s, n in docs.most_common():
        share = totals[s] / len(rows)
        item_share = len(items[s]) / flagged if flagged else 0
        out.append({"source": s, "corpus_share": 100*share, "docs": n,
                    "items": len(items[s]), "enrichment": item_share/share if share else None})
        print(f"  {s[:38]:38s} {100*share:7.1f}% {n:6d} {len(items[s]):6d} "
              f"{item_share/share:6.1f}x")
    save("composition", {"flagged": flagged, "slices": out})


# ----------------------------------------------------------------- entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "consensus", "scaling", "tuning",
                                        "mechanism", "composition", "validate", "second"])
    args = ap.parse_args()
    configure_logging()

    needs_qwen = args.stage in (None, "main", "scaling", "tuning", "composition", "second")
    qwen = _hf_tokenizer("Qwen/Qwen2-7B-Instruct") if needs_qwen else None

    stages = [args.stage] if args.stage else [
        "main", "consensus", "scaling", "tuning", "mechanism", "composition",
        "validate", "second"]
    for s in stages:
        if s == "main":
            stage_main(qwen)
        elif s == "consensus":
            stage_consensus()
        elif s == "scaling":
            stage_scaling(qwen)
        elif s == "tuning":
            stage_tuning(qwen)
        elif s == "mechanism":
            stage_mechanism()
        elif s == "composition":
            stage_composition(qwen)
        elif s == "validate":
            stage_validate()
        elif s == "second":
            stage_second(qwen)


if __name__ == "__main__":
    main()

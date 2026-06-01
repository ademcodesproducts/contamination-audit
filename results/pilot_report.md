# Pilot Report: Semantic Contamination in s1K vs MATH-500

## Key Numbers

| Metric | Value |
|--------|-------|
| Candidate pairs (from embedding retrieval) | 5,000 |
| Pairs labeled CONTAMINATED by judge | 7 |
| Pairs labeled RELATED by judge | 0 |
| Pairs labeled CLEAN by judge | 4983 |
| Errors / unparseable | 10 |
| Unique s1K items flagged CONTAMINATED | 7 |
| Unique MATH-500 items with a CONTAMINATED s1K pair | 7 |

## Interpretation

**Weak signal.** Only 7 CONTAMINATED pairs found. This is below the >20 threshold for a compelling finding. Consider: (1) lowering the cosine similarity threshold, (2) improving the judge prompt, or (3) reconsidering the dataset pair (try OpenThoughts-114K vs MATH-500).

## Similarity Distribution by Label

```
label       CLEAN  CONTAMINATED  ERROR
sim_bucket                            
<0.60        4907             2     10
0.60-0.65      55             1      0
0.65-0.70      14             0      0
0.70-0.75       6             0      0
0.75-0.80       1             0      0
0.80-0.85       0             1      0
≥0.90           0             3      0
```

## CONTAMINATED Pairs by MATH-500 Subject

```
math500_subject
Intermediate Algebra      6
Counting & Probability    1
Name: math500_id, dtype: int64
```

## Next Steps

1. **Manual validation**: Open `results/precision_worksheet.tsv`, review each pair,    fill in `human_label` and `human_notes`. Target: 20–30 cases.
2. **Compute precision**: precision = (human CONTAMINATED) / (judge CONTAMINATED) in your sample.
3. **If precision ≥ 70% and n_contaminated ≥ 20**: write up findings.
4. **If precision < 60%**: revise judge prompt (make q2/q3 stricter) and rerun.
5. **Scale up**: if pilot succeeds, run on OpenThoughts-114K for the full paper.

## Files Generated

| File | Description |
|------|-------------|
| `results/judge_results.parquet` | All judged pairs with labels |
| `results/flagged_pairs.tsv` | Human-readable CONTAMINATED+RELATED pairs |
| `results/manual_review_sample.tsv` | 30-pair sample for manual review |
| `results/precision_worksheet.tsv` | Blank worksheet for precision estimation |
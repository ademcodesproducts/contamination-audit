# Reframed Scope: Contamination Laundering → ICLR, 3 Weeks

**Status:** working scope, drafted 2 Sep 2026
**Source:** rescopes `new_proposal/rescopet.txt` and `INFO_298_Proposal_Contamination_Laundering.md` (12-week / ICML 2027) down to a 3-week ICLR submission
**Team:** Arno Demarteau, Rishbha Jain
**Target:** ICLR, deadline ~23 Sep 2026

> **Venue check — do this first.** The INFO 298 proposal lists ICML 2027 (22 Jan 2027) as primary and **ICLR 2028** as fallback, which places the ICLR cycle whose deadline lands ~3 weeks out at **ICLR 2027**, not 2026. Confirm the exact date and template cycle before anyone writes to it. This file keeps the filename it was requested under.

The 12-week proposal is the right project. This document keeps its scientific core and cuts everything that cannot be finished, verified and written in 21 days.

---

## 1. The reframe that makes 3 weeks possible

The proposal is scoped as *estimate an effect*. Rescope it as *build a detector and report AUC*. Three consequences, all of them upgrades:

### 1.1 Randomized assignment kills the confound

Split MATH-500, stratified by subject × level, into 250 injected / 250 held out, and **randomize** which side each item lands on.

The control group is then matched **by construction**. Parallel trends holds by design rather than by assertion — which matters because the ANLP draft asserts exact subject × level matching in §3.2 and the shipped data contradicts it (OpenThoughts contaminated avg level 3.50 vs clean 2.15; Tülu 3.28 vs clean 4.75, confounded in opposite directions per model). This fix is free and it is the single biggest methodological upgrade in the plan.

### 1.2 Ground truth by construction kills the judge

We know which items were injected. That removes, structurally:

- the LLM-judge definition of the treatment group (currently 28% precision on OpenThoughts C_sem, 38% on Tülu C_sem);
- the annotation backlog (7 of 24 OpenThoughts treatment items have a human verdict; 4 of those 7 are `N`);
- the unverifiable κ = 0.83 claim, which has no supporting artifact in the repo.

No annotation is on the critical path any more.

### 1.3 AUC at n=500 is well-powered where a DiD at n=24 was not

The prior paper tried to estimate a ~0.05 aggregate effect from 24 binary observations, and every CI spanned zero. A per-item classification metric with 250 positives and 250 negatives has real resolution, and — the important part — it puts membership detection and behavioral detection **on the same axis in a single figure**.

**The DiD becomes a feature the behavioral detector consumes, not the headline result.**

---

## 2. What the 3-week paper is

| Section | Content | Cost |
|---|---|---|
| §1–2 Motivation | Failure-mode taxonomy from the existing repo: s1 97% lexical (implementation bug), Tülu 66% semantic (design limit), OpenThoughts leaking 18.6% lexically despite the strictest n-gram setting of the three. Plus the Tülu source localization (all leakage in NuminaMath-TIR at 8.4%; GSM8K and PersonaHub at 0.0%). | **Already computed.** Zero marginal cost. |
| §3 Related work | Positioned directly against *Evading Data Contamination Detection for LMs is (too) Easy*, Yang et al. (2023) rephrased-samples, ConTAM. | Writing only |
| §4 Method | Controlled injection at graded laundering strength; detector suite; behavioral detector. | Week 1–2 |
| §5 Results | The evasion curve: membership AUC → 0.5 as laundering increases, behavioral AUC holding. | Week 2 |
| §6 Limitations | Dose realism, single base model, pretraining contamination. | Writing only |

§1–2 is what stops the paper reading as a purely synthetic exercise: real pipelines fail, here is the evidence, here is the controlled study that explains why.

**Do not edit the ANLP paper into this.** It is a different paper. The old work enters as motivation and related work only, and is not submitted anywhere itself.

---

## 3. Experimental design

### 3.1 Conditions — 5 checkpoints, one base model

| Checkpoint | Injected training text | Role |
|---|---|---|
| `control` | none | Specificity / false-positive rate for every detector |
| `verbatim` | benchmark item as-is | Positive control — everything must detect this |
| `light` | surface rewording, same structure | |
| `heavy` | full semantic rewrite, same underlying problem | |
| `numeric` | surface numbers changed, same solution schema | Tests whether our own perturbation detector sees through the laundering it is designed to resist |

The **same** 250/250 item split is used across all conditions. Only the laundering of the training text varies. This keeps the evasion curve a clean one-variable sweep.

### 3.2 Dose

Inject at a dose high enough that the verbatim condition definitely took — repeat contaminated items 4–8× in the mixture, or train additional epochs. Verify before moving on.

State plainly in the paper that this measures **detectability under a known effect**, not a real-world contamination rate. It is an upper bound on what detectors can find: if they fail here, they fail everywhere. Frame it before a reviewer frames it for us.

### 3.3 Inference volume

```
5 checkpoints × 500 items × {original, number_swap} × 5 samples  = 25,000
+ surface_noise specificity check on control + verbatim × 5      =  5,000
                                                          total ≈ 30,000 generations @ 4,096 tok
```

Roughly 2–3 days on the Blackwell **with vLLM**. Not achievable in three weeks without it — see §4.

### 3.4 Metrics

- **Primary:** detection AUC per method per laundering strength. One figure, two curves.
- **Secondary:** DiD estimate and bootstrap CI per condition; CoT features (self-correction, null rate, hedging, math-token density) per condition.
- **Pre-registered**, written before week-2 results are examined, included as an appendix. Cheap, and it inoculates us against the post-hoc-null-rate-rescue critique the ANLP draft is wide open to.

---

## 4. Engineering prerequisites — day one, before anything else

1. **Move to vLLM.** `src/contamination_audit/inference.py:114` calls `model.generate` one prompt at a time. At ~30,000 generations that is the entire three weeks. With continuous batching it is two or three days. **Nothing else in this plan matters if this does not happen first.**
2. **Add `peft`.** No LoRA anywhere in the repo yet. Dry-run the full inject → fine-tune → checkpoint → eval loop on 20 items before trusting it with a real condition.
3. **Land the real perturbation generator.** `src/contamination_audit/perturbation.py:31` is a `NotImplementedError` stub; the stimuli used in the ANLP paper are not reproducible from the repo. The laundering pipeline needs this code anyway, so it is not extra work.

---

## 5. Cut list

| Cut | Why |
|---|---|
| Code / cross-domain extension (HumanEval, LiveCodeBench) | Doubles every pipeline stage for one additional data point |
| GRPO / RL post-training ablation | A project in its own right — this is the ICML 2027 contribution |
| Multiple base models | One base model, stated as a limitation |
| Mid-semester presentation deliverable | Irrelevant to a 3-week sprint |
| Further annotation of the observational treatment sets | Injection supersedes it; no longer on the critical path |

**Keep Min-K%.** It is cheap (logprobs only) and it is the black-box baseline reviewers expect. Cutting it invites *"you only broke the weak detectors."*

---

## 6. Week by week

### Week 1 (Sep 2–8) — de-risk the hypothesis, not build the pipeline

The proposal's ordering (build everything → fine-tune → detect) means we would learn whether the paper works in week 10. In a 3-week sprint we must know by day 5.

- Build the injection harness and the stratified randomized 250/250 split.
- Swap inference to vLLM; add `peft`; dry-run the loop on 20 items.
- Fine-tune **only `control` and `verbatim`** — verbatim needs no paraphrase generation and is where the effect must exist if it exists at all.
- Run N=5 on both checkpoints; compute both detector families.
- **In parallel, off the critical path:** paraphrase generation for `light` / `heavy` / `numeric` (API calls, no GPU).

> ### GATE — approx. Sep 6–7
> **Does behavioral detection separate injected from control under verbatim injection?**
> **Yes →** everything after this is execution.
> **No →** the central hypothesis is dead and we need the remaining two weeks to reframe (see §8).

### Week 2 (Sep 9–15) — sweep and measure

- Fine-tune `light`, `heavy`, `numeric`.
- Full inference sweep across all five checkpoints.
- Run all detectors: n-gram, embedding retrieval, Min-K%, LLM-judge, behavioral.
- Build the evasion curve.
- **Freeze numbers Friday 15 Sep. No exceptions** — unfrozen numbers are what produced the paper/repo divergence in the last submission, where not one PDF number is reproducible from the repository.

### Week 3 (Sep 16–23) — write and release

- Paper drafted from scratch against the §2 section map.
- Pre-registration appendix included.
- Release: injection pipeline, labeled checkpoints, detector suite, reproduction instructions where every paper number maps to one command.
- Anonymized build compiles from a fresh checkout **on a machine that is not ours**.
- Internal read-through Sep 21; submit Sep 22 with a day of margin.

---

## 7. Two objections reviewers will raise

**"Is this dose realistic?"** No, and we say so first. See §3.2 — it is an upper bound on detectability, deliberately.

**"Your base model may have pretrained on MATH-500."** It may have. Under the old observational design that was a confound; under injection it shifts every condition equally, so it moves the level and not the shape of the evasion curve. This is a genuine advantage of the new design and is worth an explicit sentence.

---

## 8. Fallback if the gate fails

There is still a paper: *all* detection methods, including causal-behavioral ones, collapse under laundering — a stronger, post-training-specific extension of *Evading Data Contamination Detection is (too) Easy*, shipped with a released benchmark and labeled checkpoints for the community. Datasets & Benchmarks track is a viable home.

**Draft this framing in week 1** so a bad gate on day 5 costs a paragraph rather than the submission.

---

## 9. Team split

Week 1 must run in parallel or it does not fit.

- **Arno:** infrastructure — vLLM swap, LoRA harness, injection split builder, inference sweep.
- **Rishbha:** detectors — n-gram, embedding, Min-K%, LLM-judge — plus paraphrase generation for the three laundering conditions.
- **Both:** week 3 writing.

---

## 10. Carry-over hygiene from the ANLP repo audit

Cheap, and each one removes a reason for a reviewer to stop trusting the rest. Do them in week 1, not week 3.

- [ ] Comment out `\iclrfinalcopy` (`paper/iclr2026_conference.tex:28`) — currently active with all three author names, which breaks double-blind.
- [ ] Add `references.bib`, `.sty`, `.bst`; confirm the paper compiles at all.
- [ ] Fix `lex_share_pct` + `sem_share_pct` summing to 109.7% in `results/tables/failure_mode_comparison.csv`, and the 27.0% vs 24.6% mismatch in the matching `.txt`. Add a regression test. **These numbers appear in §2 of the new paper**, so this is load-bearing, not cosmetic.
- [ ] Untrack `results/traces/` from `.gitignore` so the release can reproduce its own tables.
- [ ] Delete `REFACTOR_NOTES.md` before the public push.
- [ ] Resolve which judge produced the old Tables 4–5 (README open question: §5.4 says GPT-4o-mini, the teammate script defaults to Gemini 2.5 Flash) — only matters if any of those numbers survive into §2.

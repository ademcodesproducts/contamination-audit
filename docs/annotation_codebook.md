# Contamination Annotation Codebook

You are labelling **pairs**: one MATH-500 benchmark problem, and one training item
retrieved from a released training corpus. You decide what relationship the pair has.

Two people label every pair independently. Do not discuss a pair before both of you
have recorded a label. Disagreements are resolved later in adjudication, and the
disagreement rate is itself a reported result — so a disagreement is data, not a
mistake.

## The one question

> **If a model memorised the training item, would that alone let it produce the
> correct answer to the benchmark problem?**

Everything below is a way of making that question checkable. When a case is unclear,
return to this sentence and answer it literally.

Note what this question excludes. A training item can be *about the same topic*, use
the *same technique*, or come from the *same source book* and still be a NO. Learning
a method is what training is for. The question is specifically about obtaining **this
problem's answer** without doing this problem's work.

## Labels

| Label | Meaning | Test |
|---|---|---|
| `VERBATIM` | Same problem, essentially the same wording | Could you mistake one for a copy of the other? |
| `PARAPHRASE` | Same problem, reworded, **same answer** | Different words, same quantities, same final answer |
| `ANSWER_LEAK` | Training item states this problem's answer | The answer appears, even without the full problem |
| `TEMPLATE` | Same structure, **different numbers → different answer** | Solving it does not give you this answer |
| `SAME_SKILL` | Same technique, different problem | Teaches the method, not the answer |
| `UNRELATED` | No meaningful relationship | — |
| `UNSURE` | You genuinely cannot tell | Use it; do not guess |

**Contaminated** = `VERBATIM`, `PARAPHRASE`, or `ANSWER_LEAK`.
**Not contaminated** = `TEMPLATE`, `SAME_SKILL`, `UNRELATED`.

`UNSURE` is a real option. An honest `UNSURE` rate is more useful to this paper than
a confident coin flip, because the paper's claim is partly that these judgements are
hard. Do not use it to avoid thinking — use it when two labels are genuinely defensible.

## The boundary that matters most

`PARAPHRASE` vs `TEMPLATE` is where nearly all disagreement lives, and it is the single
most important distinction in this codebook.

- Numbers identical, wording different → **PARAPHRASE** (contaminated). Memorising it
  hands you the answer.
- Numbers different, so the answer differs → **TEMPLATE** (not contaminated). Memorising
  it hands you a method you still have to apply.

Check the **final answer**, not the surface text. Two problems can read almost
identically and have different answers; two can read differently and have the same one.
The answer is what decides it.

If the numbers differ but the answer coincidentally lands on the same value, label
`TEMPLATE` and flag it in the notes column.

## Procedure

Work through these in order and stop at the first that fires:

1. **Read the benchmark problem and note its final answer.** Do this before reading the
   training item, so the training item cannot anchor you.
2. **Scan the training item for that answer.** If it appears as the answer to a matching
   question → `ANSWER_LEAK`.
3. **Compare the problem statements.** Near-identical → `VERBATIM`.
4. **Compare the quantities.** Same quantities and same answer, different wording →
   `PARAPHRASE`.
5. **Different quantities?** → `TEMPLATE`.
6. **Different problem, same method?** → `SAME_SKILL`.
7. **Otherwise** → `UNRELATED`.

## What to ignore

These are not evidence either way. Retrieval surfaced the pair; that is not a signal.

- **Similarity scores.** If a score column is visible, ignore it. It is the thing being
  evaluated, so letting it steer you contaminates the measurement.
- **LaTeX and formatting differences.** `\frac{1}{2}` and `1/2` are the same quantity.
- **Length.** A long worked solution and a terse one can be the same problem.
- **Topic and difficulty overlap.** MATH-500 is standard competition material; heavy
  topical overlap with any math corpus is expected and is not contamination.
- **Whether you think the corpus "should" be clean.** Judge the pair in front of you.

## Recording

For every pair record: `label`, `confidence` (high / medium / low), and `notes`.

Write a note whenever you hit a boundary case, and always for `UNSURE`. One line is
enough — "same setup, radius differs, answer differs" is a good note. These notes drive
adjudication, and boundary cases are where this paper's argument is made.

Do not revise earlier labels after seeing later pairs. If your reading of the boundary
shifts partway through, say so in the log rather than going back — a drift you record
is analysable; a silent one is not.

## Adjudication

After both passes are complete:

1. Compute agreement before any discussion. This is the reported number.
2. Both annotators review disagreements together and record an agreed label, plus one
   line on why the disagreement happened.
3. Pairs still unresolved stay `UNSURE` and are reported as such. Do not force them.

The pre-discussion agreement is what gets published. The adjudicated labels are the
gold standard everything else is scored against.

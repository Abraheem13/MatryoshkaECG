# Matryoshka-ECG — revision package

Response to the major-revision decision. Contains the restructured manuscript,
a point-by-point response letter, and the code implementing every experiment
the reviewers requested.

**Start here:** [`RUNBOOK.md`](RUNBOOK.md) — environment, data, commands, timings.

```bash
cd code && pip install -r requirements.txt
python scripts/smoke_test.py --full     # ~2 min, no data needed, validates everything
```

---

## Read this first

The manuscript contains **no results yet**. Every number is a macro rendering
as a red **[TBD]** until you run the pipeline on your GPU.

I could not run these experiments — no GPU, no PTB-XL, no checkpoints where
this was prepared. Writing plausible numbers into a revision that responds to
reviewers objecting to *unsupported claims* would be self-defeating, and the
real risk is that one survives to submission. So results flow one way only:
experiments → `aggregate_results.py` → `generated/*.tex` → paper. `make check`
fails the build while any placeholder remains.

What is complete and ready: all argumentation, framing, statistical
methodology, the response letter, and code that has been syntax-verified,
unit-tested, and dry-run end to end on synthetic data.

---

## Reviewer comment → what changed

| # | Comment | Change | Where |
|---|---|---|---|
| R1-1 | Wearable claim unsupported by PTB-XL | Claim withdrawn; retitled; new "what this paper does not claim" section; cross-dataset + reduced-lead + artefact experiments added | §I-A, §V-E, `eval_transfer.py`, `preprocess_external.py` |
| R1-2a | Latency/memory/energy not established | Real measurement (CUDA events, NVML energy, CPU, INT8, ONNX); extrapolated table deleted; claim narrowed to storage + artefact count | §V-C, `benchmark_hardware.py` |
| R1-2b | Missing fixed Inception1D baseline | Added at 6 dims × 5 seeds, both backbones; previous cross-architecture comparison identified as confounded | §V-B, `run_all.sh` Stage 3 |
| R1-2c | No repeated runs or statistics | 5 seeds; bootstrap CIs; DeLong; Holm; **equivalence test** with pre-registered margin | §IV-E, `analysis/stats.py` |
| R1-m1 | Imprecise terminology | Three-way distinction (deployment mapping / on-device inference / wearable validation) made structural | throughout |
| R2-1 | Doesn't beat stronger classifiers | Stated plainly; superiority framing dropped | §I, §V-G |
| R2-2 | Missing fixed Inception1D | see R1-2b | |
| R2-3 | Deployment not hardware-validated | see R1-2a | |
| R2-4 | Clinical hierarchy speculative | Ridge probing of 11 physiological descriptors; paper commits to withdrawing Table II if unsupported | §V-D, `analysis/probing.py`, `features.py` |
| R2-5 | Limited to PTB-XL 5-class | CPSC-2018 / Chapman / Georgia transfer; finer granularity explained as a *test of the mechanism*, flagged not fudged | §V-E1, §VI-D |
| R2-6 | No uncertainty analysis | see R1-2c | |

### Extensions added because compute is not your constraint

| Addition | Why it earns its GPU time |
|---|---|
| 23-class diagnostic task (`TASK=sub`) | The sharpest open prediction in the paper: if flatness comes from low intrinsic dimensionality, the sufficient prefix must *grow* with task granularity. Tests the mechanism rather than extending coverage. |
| Native reduced-lead training (Stage 5) | Trains directly on single-lead input rather than masking a 12-lead model — the closest available proxy to the wearable claim R1 rejected. |
| Quantisation (Stage 9) | Measures the compute axis nesting leaves untouched, and whether INT8 penalises *small* prefixes disproportionately. That interaction appears unreported for nested representations in any domain. |
| XResNet1D-50, InceptionTime-faithful variants | Widens the architecture-agnosticism evidence beyond two points. |

### Issues found while revising (not raised by reviewers)

1. **Per-record normalisation destroyed absolute voltage** — the diagnostic
   criterion for hypertrophy. HYP was the worst class by a wide margin
   (0.811 vs 0.93). Likely also part of the gap to the published benchmark.
2. **Weight decay applied to BatchNorm** — `'bn' in name` missed every
   normalisation layer nested inside `ConvBlock1d`.
3. **F1 at fixed 0.5 threshold** — explains most of the AUC 0.90 / F1 0.62 gap.
4. **Paper–code mismatch** — SE attention described but not implemented; also
   XResNet1D cited to He et al. and Inception1D to Szegedy et al. rather than
   Strodthoff et al. and InceptionTime.
5. **SVD baseline leakage** — basis and classifier fitted on the test split,
   then plotted as an upper bound against honestly evaluated models.
6. **Model selection on the largest head** — biased the checkpoint toward one
   operating point, which is what a nested model should avoid.
7. **Figure axis scaling** — a 0.02-AUC window magnified sub-0.005 differences
   into apparent structure.

---

## The two changes that matter most

**The compute argument was wrong in kind, not degree.** Reviewer 1 is right
that the backbone dominates, and this is intrinsic: prefix truncation happens
*after* the encoder runs, so latency is flat in *d* by construction. No
measurement could rescue the original claim. The paper now says this plainly
and marks its own method "No" in the compute-saving column of the related-work
table. The real benefits — 32× embedding storage, one artefact instead of six,
one validation pathway — are stated precisely.

**The headline baseline comparison was confounded.** "Outperforms six
independently trained 34M baselines" compared Inception1D-MRL against
XResNet1D-101 fixed models. Since Inception1D also beat XResNet1D-101 *under
MRL*, that comparison confounds nesting with backbone choice. The matched
baselines separate them, and the regulariser interpretation — which rested on
the same confound — has been removed.

---

## Verification performed

Every path was **executed**, not merely syntax-checked. Torch was installed and
the pipeline run end to end on synthetic data.

- `smoke_test.py` — 20+ checks green: all five backbone/head combinations,
  prefix-truncation equivalence, eval-loader alignment, both normalisation
  modes, artefact injection determinism, full training runs (MRL, MRL-E,
  fixed-dim, reduced-lead, per-record ablation).
- `tests/test_stats.py` — 8/8, with `fast_auc` and DeLong verified against
  sklearn to 1e-9.
- Executed end to end: `train.py`, `eval_transfer.py` (svd / leads / external /
  robustness, 26 conditions), `quantize.py`, `benchmark_hardware.py`,
  `run_probing.py`, `aggregate_results.py`, `make_figures.py`.
- `main.tex` (10 pp) and `response_to_reviewers.tex` (5 pp) compile clean, no
  undefined references or citations.
- Placeholder gate (`make check`) confirmed to fail while `[TBD]` remains.

**Three bugs were caught by running the code that reading it had missed** — the
worst being an `nn.ModuleDict` integer-vs-string key lookup that made *every*
MRL forward pass raise. That is the argument for running `smoke_test.py`
before you commit GPU hours.

## Before submitting

1. Run the pipeline (`RUNBOOK.md` §2).
2. `cd paper && make check` — must report `OK: no placeholders remain.`
3. Resolve the outcome-dependent passages in §V-D and the `$\dagger$` markers
   in the response letter once the probing verdict is known.
4. Fill author names, affiliations, and funding in `main.tex`.
5. Re-read §VI-C — if the matched baselines show MRL genuinely *beating*
   same-backbone fixed training, the regularisation account becomes defensible
   and that paragraph should be strengthened rather than hedged.

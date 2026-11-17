# Computational modernization of classical statistical models: research roadmap

**Version** 0.9, draft for the maintainer's review · **Date** 2026-11-14 (version 0.1: 2026-11-13)

> **Versions 0.4 and 0.5 follow the maintainer's instructions of 2026-11-14 on the workspace** (0.5 corrects the list of reference packages). The work moves to a new, independent repository, `statcompute`, where this document now lives. pprof (R), pprof_py and pprof_spark are read-only references: track A is implemented in the new repository's engine, not in pprof_spark. The standard cluster is a 32 GB driver, 10 workers of 16 GB (confirmed 2026-11-14; version 0.6) and Databricks Runtime 17.3 LTS (Apache Spark 4.0.0). Decisions 5 and 7 and F-1's preconditions change accordingly; the review and the opportunities do not. Where this document says "the repository" in sections 2 to 7, it means pprof_spark, the reference that was read.

> **Version 0.3 uses the six rows of `scale-first-look.csv` that the maintainer supplied** (2026-11-14). F-1 is rewritten around them. Summary items 3 and 8, step 0 in section 8.2 and decision 5 change with it. Nothing else does.

> **Version 0.2 records the maintainer's answer to decision 5** (2026-11-14): the slow Cox run was stratified by 8,000 providers. F-1, the note under the table of section 7 and decision 5 change, with two sentences of the summary. Nothing else does.

**Sources.** The project brief; the feasibility study `algorithmic-feasibility-study.md` of 2026-11-13 ("the study"); pprof_spark at `e5c9484` (round 146; the study read `a28a224`, round 139), pprof_py at `94b5476` and R pprof 2.0.12 at `d780cf5`; H2O-3's `CoxPH.java` on its master branch; the `SQLConf.scala` of Spark 3.5.2, 4.0.0 and 4.1.0, for three defaults; the six result rows named above; the references of Appendix B. Seven sandbox probes on synthetic data support the statements marked as such. Their output is in `roadmap-evidence.md` and their source in `roadmap-probes.tar.gz`. Nothing was implemented in any package, no cluster was used, and no protected data were seen.

**Evidence tags.** Every statement that is not the study's or the brief's carries one of these.

| Tag | Meaning |
|---|---|
| [study] | Taken from the feasibility study; its own class (read, measured, derived) applies |
| [repo] | Read in the repositories at the commits above |
| [probe A-n] | Measured in a sandbox probe of Appendix A: one core, toy data, NumPy or one JVM |
| [derived] | Arithmetic or algebra done here from stated figures |
| [read in source] | Read in a third-party source file named in the text: H2O-3 or Spark 4.1.0 |
| [rows] | Read in the six rows of `scale-first-look.csv` supplied on 2026-11-14 |
| [hypothesis] | Proposed and not yet tested |
| † | A reference or product fact checked against a publisher, index, vendor or arXiv record on 2026-11-13 or 14 |

**Contents.** 1. Summary · 2. Review of the feasibility study · 3. Research vision · 4. Research opportunities · 5. Literature landscape · 6. The two case studies as starting projects · 7. Prioritization · 8. Development roadmap · 9. Decisions requested · Appendix A. Sandbox probes · Appendix B. References added by this document

## 1. Summary

1. **The study settles the first question for both case studies, and I agree with it.** Keep Newton's method with an explicit information matrix, and change how that matrix is accumulated and reduced. I recomputed the study's arithmetic, derived its two identities independently and reproduced its qualitative findings on other synthetic data (section 2.1). Nothing I found changes a recommendation or the order of the recommendations.

2. **I propose eleven amendments, F-1 to F-11 (section 2.3).** Four affect specifications:
   - centring covariates per stratum in the weighted-Gram form, without which that form was 12 to 120 times less accurate than the risk-set form when stratum means were spread widely [probe A-7];
   - the Efron and entry-time bookkeeping of the per-time layout;
   - the equations that define the estimate when providers sit at the bound;
   - a tolerance class for converged parity.

3. **The largest open risk is the study's own last paragraph, and the recorded runs sharpen it.** CPU per row does not explain a Cox run on the cluster that is no faster than one pprof_py thread. The rows the maintainer supplied show a build 68 commits behind main, a Spark 3.5.2 runtime, and 2 to 10 small workers whose number changed between runs. Three of the six rows are not completed fits. The three that are used the equivalent of 2 to 6 cores. Step 0 has to run under controlled conditions before any later step is credited with a wall-clock gain (F-1, F-2).

4. **Inference is the unexamined half.** The study covers fitting. By the repository's own complexity estimates, and under the assumptions stated in F-2, the exact provider limits alone cost the same order of CPU as a vectorized fit. One gain is already checked: with exponential tilting the limits need one recursion per provider instead of one per trial value (O-5) [probe A-2].

5. **Research novelty lies mostly beyond the two case studies.** For the logistic model the study shows that no structure is left to find. For Cox, the per-time layout without second moments is the one candidate for a methods contribution, and H2O-3 already aggregates on a time grid, with second moments (F-3). The open problems have two separable blocks instead of one: Cox with provider effects, crossed provider and cluster terms, and the pass counts of random-effect fits (section 4).

6. **The roadmap has four tracks (section 8).** Track A is the study's steps 0 to 6, in the study's order, with the amendments attached. Track B is exact inference. Track C is the second-wave structures. Track D is the research register and the publications.

7. **Expect constant factors, not new complexity classes.** The study derives about 8 times less CPU per Cox fit and 4 to 6 times less per logistic fit. Those are engineering results, and they are worth having. Section 3.4 lists what could be published and how new each item is likely to be.

8. **Seven decisions are requested (section 9).** Two would narrow the work most: aggregate facts about the production designs, and the benchmark environment.

## 2. Review of the feasibility study

### 2.1 What I checked

| Study | Claim | How I checked it | Result |
|---|---|---|---|
| §2 | Working sets of 7.3, 29 and 146 GB; a packed information matrix of 131 KB | Recomputed from 8·n·(p + 3) bytes and from p(p + 1)/2 + p + 2 doubles | Agree [derived] |
| §3 | While every block's partial is collected, blocks must grow with n·p³ | From the driver budget: at most budget/(4p²) blocks, so at least 4n·p²/budget rows of 8(p + 4) bytes per block | Agree [derived] |
| §4.1 | Information = Xᵀ·diag(rᵢcᵢ)·X − Σₜ dₜ·x̄(t)x̄(t)ᵀ, with the stated Efron and entry-time forms | Derived independently from the Poisson form of the Cox model; then a probe with Efron ties, entry times, case weights, offsets and five strata, against explicit risk sets and central differences of the score | Agree to 1.2·10⁻¹⁵ (Breslow) and 7.8·10⁻¹⁶ (Efron) of the largest entry [probe A-1]. These are the cases the study derived and did not run |
| §4.2 | 2.7 MB to the driver where second moments per time would need 238 MB | 1,825 × 182 × 8 bytes and 1,825 × 16,290 × 8 bytes | Agree [derived] |
| §4.3 | Profile-likelihood Newton reaches SerBIN's optimum in fewer passes | A probe on 207,330 rows, 1,994 providers and 20 features, none at the bound | 4 profile steps against 6 joint steps; β agrees to 2·10⁻¹² [probe A-6]. The bounded providers were not replicated |
| §5.1 | The relative change in ℓ stops a chord iteration early | A probe with the information frozen after the first step | Reproduced: it stopped with a largest score component of 0.26, 2·10⁻⁵ from the root [probe A-5] |
| §5.2 | The cost table and the figures at 20 million rows | Rebuilt each row from the four measured rates, 1 μs of decode per pass and the pass counts | Reproduce to within 2%: 312 and 38 μs per row for Cox; 237, 60 and 39 for the logistic model; 6,240 to 760 and 4,740 to 780 CPU-seconds [derived] |
| §5.2 | First-order methods cross over between 400 and 1,000 features | Solved 6·(1.7·10⁻⁴·p²) = N·(10⁻²·p) for N = 45 and 100 | p = 441 and 980 [derived] |
| §4.4, §5.2 | The JVM rates | Not reproducible here: I did not build the engine. A probe times a simpler kernel of the same kind | Same direction: chunked accumulation 2.4 to 2.5 times faster than per-element Neumaier sums at p = 50 and 100 [probe A-4]. The study's larger factors are plausible, because the engine's kernels do more per element than one compensated addition |

One reading note for §5.2: "rounds" counts every pass of a fit. The 12 rounds of the present logistic fit are five steps of two passes plus the variance and co-moment passes, and the 6 of the profile method are five information passes plus the variance pass. That is how the table's figures reproduce.

### 2.2 Assessment

The study's method is sound. It separates what was read, measured and derived; it keeps the rejected candidates with their numbers; and it states its limits. Its recommendations follow from its evidence, and its order of work is right: changes that keep today's iterates come first, because they can be verified against the existing fixtures at every level.

The study's §9 lists three things assessed by argument and not by a run. The probes here reduce two of them at the level of the mathematics, not of the engine:

- Efron ties and entry times in the weighted-Gram form [probe A-1];
- sparse covariates [probe A-4, a first number only].

The third, providers at the bound with weights or trials, stays open.

### 2.3 Amendments

**F-1. Make step 0 explain wall-clock time.** By the study's rates, a pprof_spark Cox evaluation costs three times pprof_py's CPU per row (51 against 17 μs), yet the maintainer saw a Cox run on the cluster that was no faster than one pprof_py thread. Kernel CPU cannot be the main term of such a run. On 2026-11-14 the maintainer supplied the six rows of `scale-first-look.csv`; all have 8,000 providers, and the Cox fits were stratified by provider.

*The runs were not what the study and version 0.1 assumed.*

| Assumed | Recorded [rows] |
|---|---|
| 30 workers of 32 GB | Workers of type m5dn.xlarge, which has 4 vCPUs and 16 GiB†. Between 8 and 40 cores were attached when each row was written, so 2 to 10 workers. The `workers` field is 1 in every row |
| A cluster of fixed size | The attached cores differ in every row, and data generation per reported core varies fourfold between rows [derived]. The cluster changed size between runs, and probably within them |
| Current main | Build `b670dbb`, round 81: 68 commits behind main, and before round 83 let blocks grow to fit the driver budget [repo] |
| Spark 4.1 on Databricks Runtime 18 | Runtime 16.4 with Photon, which is Apache Spark 3.5.2†. The repository's guide asks for Runtime 18 and a fixed-size cluster [repo] |

*Three of the six rows are not completed fits.*

| Row | Run | Blocks at that build's defaults | Driver-budget guard | What the row holds |
|---:|---|---:|---|---|
| 1 | Logistic, 5 million × 100 | 992 | Passes | Its own fit, 111.56 s |
| 2 | Logistic, 20 million × 100 | 3,968 | Rejects | Row 1's iterations and largest z; no fit time |
| 3 | Logistic, 8 million × 100 | 1,587 | Passes, 3% under the limit | Its own fit, 267.13 s |
| 4 | Logistic, 10 million × 100 | 1,984 | Rejects | Row 3's values, to the last digit |
| 5 | Cox, 10 million × 100 | 1,984 | Rejects at the defaults | Its own fit, 165.27 s, so its block options were presumably raised |
| 6 | Cox, 5 million × 180 | 1,756 | Rejects | No fit time; a largest z of 13.1, which is row 5's 100 coefficients set against the truth for 180 features |

The guard admits 1,628 partials at 100 features and 509 at 180 [derived from `LayoutPlan.create` and the guards at `b670dbb`]. The notebook's last cell writes a row from whatever `fit` and `timings` the session holds, so after a failed fit it records the previous one [repo]. These rows therefore hold no timing of a Cox fit at 180 features.

*The three completed fits used few cores.*

| Fit | Cores reported | Fit, s | Kernel CPU, s | Effective cores |
|---|---:|---:|---:|---|
| Logistic, 5 million × 100 | 20 | 111.56 | 335 | 3.0 to 4.8 |
| Logistic, 8 million × 100 | 16 | 267.13 | 535 | 2.0 to 3.8 |
| Cox, 10 million × 100 | 40 | 165.27 | 703 | 4.3 to 6.0 |

Kernel CPU scales the study's rates to 100 features: 17 μs per row for a Cox evaluation, and 67 μs per row for a four-step logistic fit with its variance and co-moment passes [derived; the study measured at 50 and 180 features, on another CPU]. Effective cores are kernel CPU over wall time. The lower figure uses the whole fit. The higher one first subtracts the downstream call's time, as an estimate of preparation, because that call rebuilds the working set. A vCPU of these workers is a hyperthread, so the reported cores overstate capacity.

The Cox fit took 165 s where one pprof_py thread would need about 210 s for its four evaluations [derived]. So the observation behind the study's last paragraph holds, but for at most 10 small workers and for a build 68 commits old.

*Explanations to test, in this order.*

1. **The cluster was smaller during the fits than at their end.** Under autoscaling, workers arrive while a job runs, and they arrive without the cached blocks.
2. **The cached working set had few partitions.** Spark 3.5.2 lets adaptive execution change a cached plan's partitioning by default; Spark 4.0.0 and 4.1.0 do not [read in source]. Adaptive execution sizes the merged partitions from the cluster's parallelism at the time the working set is built. On a cluster that is still scaling up, every later pass would then run that few tasks [hypothesis]. The number of tasks per evaluation job, in the Spark UI of those runs, settles it.
3. **Preparation.** The downstream calls, each of which rebuilds the working set, took 42, 127 and 47 s [rows].

*What the intended path should cost.* For current main on Spark 4.1 defaults, take 5 million rows, 180 features and 8,000 equal providers, the study's representative size.

| Quantity | Value | Basis |
|---|---|---|
| Blocks | 445, of 18 providers and 11,250 rows each | The driver budget admits 509 partials at p = 180, so the block size doubles twice, to 16 MiB [derived from `BudgetedLayout` and `BlockOptions`] |
| Kernel and decode per pass | 260 CPU-seconds | 5·10⁶ rows × 52 μs [study §5.2] |
| Tasks per pass | 200 by default | The shuffle-partition count when the working set was built [repo, Cox tutorial §5.7] |
| Fullest task | About 7 blocks, against 2.5 on average | 445 blocks hashed into 200 partitions; about 22 partitions stay empty [derived] |
| Wall time per pass | About 4 to 5 s on 120 or more cores | The fullest task: 7 blocks × 11,250 rows × 52 μs |
| One pprof_py thread | 85 s per evaluation | 5·10⁶ rows × 17 μs [study §3] |

Six passes should then take about half a minute, against more than eight minutes for six pprof_py evaluations. A rerun that falls far short of this has lost its time outside the kernel. Two further places to look are the wide input, since Spark turns whole-stage code generation off above 100 columns by default (`spark.sql.codegen.maxFields`) [read in source], and memory management, since every task allocates a 16 MB covariate array per block on every pass, or 65 MB at 20 million rows [derived].

*Action.* Three preconditions for step 0:

- a fixed-size cluster and current main, with the executor count recorded at the start and at the end of each fit;
- a harness that records only the fit it has just timed, and writes a failure record otherwise;
- the standard cluster confirmed. It is now stated as 10 workers of 16 GB on Runtime 17.3, where the study's §2 assumes 30 workers of 32 GB (decision 5).

The `statcompute` harness meets the first two by construction: every record compares the executors before and after its fit, and is built from nothing for each run.

Then add four items to step 0's exit measurement:

1. effective parallelism per pass, as executor CPU time ÷ (wall time × executor cores);
2. the straggler ratio per pass;
3. the shares of wall time in preparation, passes, driver work and downstream calls;
4. tasks and blocks per pass.

If the kernels account for less than half of wall time, the dominant term is fixed before step 2.

*First recorded cluster run, 2026-11-16 (version 0.7).* On Runtime 17.3 with Photon and autoscaling, at 5 million rows and 50 and 100 features, pprof_spark's time did not follow the number of executors (a logistic fit took 112 s with 2 and 149 s with 9 to 10), it used 8 to 20 of 40 attached cores, and its executor CPU was 4 to 7 times what the study's kernel rates give for the passes [derived]. Explanation 3 therefore carries most of the weight at this size, and explanation 2 does not apply on Spark 4.0. The analysis is in the `statcompute` repository, `docs/reports/2026-11-16-databricks-compare-5m.md`.

**F-2. Count the whole job, not only the fit.** The study's cost table excludes preparation and everything after the fit. The repository's own estimates suggest those parts are not small [repo, logistic tutorial chapter 8]:

- the likelihood-ratio and score tests refit the model once per tested covariate, after a rebuild;
- the exact limits take 40 to 80 tail evaluations per provider, each O(nⱼ·min(oⱼ, nⱼ − oⱼ));
- each of the covariate tests, the provider tests and the standardization rebuilds the working set.

For scale: with 8,000 equal providers of 2,500 rows and 20% events, 60 evaluations cost 6·10¹¹ inner steps. At an assumed 1 ns per step that is 600 CPU-seconds, against 780 for the vectorized fit at 20 million rows [derived; the 1 ns is an assumption]. Skewed provider sizes make it worse, because the cost is quadratic in nⱼ.

*Action.* Step 0 also reports totals per job, for the job runners with every output requested. Inference enters the roadmap as track B.

**F-3. Scope the word "new" in §4.2.** The closest prior art I found is H2O-3's `CoxPH` [read in source]. Its workers aggregate risk-set sums on a time grid, and the driver takes reverse cumulative sums. Its Breslow path allocates a time × p × p array per stratum, and for start–stop rows it loops over every grid time a row spans. The study's layout differs by dropping the second moments from the grid, through an identity that is the information of the Poisson form of the Cox model (Whitehead 1980†). I found no source that states the two-pass form of a time grid of first moments followed by a weighted Gram matrix, and my search was not exhaustive.

*Action.* Write "new to the sources checked", cite H2O-3 as prior art, and run a targeted search before any claim of novelty in a paper.

*Implemented, 2026-11-17 (version 0.9).* The `statcompute` engine fits Cox regression without strata in this layout, with Efron ties and entry times on the same grid: two passes per evaluation, no shuffle, p + 1 numbers per time. Efron's steps need three weights per time, not a vector per step, because every step's mean is a combination of the same two vectors. It agrees with explicit risk sets and with pprof_py [engine tests]. The targeted search has not been run; nothing is claimed as new. Candidates to check: Cyclops and its GPU scans (Yang et al. 2024†), the `survival` package's counting-process code, and SAS's fast option for counting-process data.

**F-4. Centre covariates per stratum in the weighted-Gram form.** The form subtracts two matrices, G = Xᵀ·diag(rᵢcᵢ)·X and M = Σₜ dₜ·x̄(t)x̄(t)ᵀ. Their difference is the within-risk-set variation. Each of them also contains the variation of the stratum means, when covariates are centred at their overall means. The partial likelihood, the score and the information are exactly invariant to a stratum-specific shift of a covariate, so centring per stratum removes that common part at no statistical cost.

Probe A-7 moved one covariate's stratum means apart and measured the relative error of the information against a reference [probe A-7]:

| Spread of stratum means, in within-stratum SDs | Risk-set form as swept today, overall centring | Weighted Gram, overall centring | Weighted Gram, per-stratum centring |
|---:|---:|---:|---:|
| 1 | 5·10⁻¹⁶ | 2·10⁻¹⁵ | 5·10⁻¹⁶ |
| 10 | 1·10⁻¹⁵ | 1·10⁻¹³ | 3·10⁻¹⁵ |
| 100 | 2·10⁻¹³ | 2·10⁻¹² | 4·10⁻¹⁵ |
| 1,000 | 5·10⁻¹¹ | 8·10⁻¹⁰ | 3·10⁻¹⁵ |

With overall centring the Gram form fails T-fn's 10⁻¹² at a spread of 100, and from a spread of 10 it is 12 to 120 times less accurate than the risk-set form. With per-stratum centring it stays at rounding level. The probe uses plain NumPy sums; compensated sums would lower every figure, but not the cancellation in G − M, which depends on the size of G relative to the difference.

*Action.* Step 3's specification centres each stratum's covariates at a reference fixed by the stratum's data, and back-transforms the reported baselines and linear predictors as §7.3.2 of the project context already requires. Add a fixture whose stratum means are spread by 100 within-stratum SDs, with a negative control. This also answers part of OI-66. A covariate that is constant within every stratum then aliases exactly.

**F-5. State the per-time layout for Efron ties and entry times.** §4.2 describes Breslow ties without entry times. The general case needs, per distinct time:

- the tied-death sums S₀ᴰ(t) and S₁ᴰ(t), for Efron's steps;
- the entry-time contributions, as negative increments on the grid.

That is about 2(p + 2) numbers per time instead of p + 2. Each row's weight is then

```text
Eᵢ = rᵢ·[c(exitᵢ) − c(entryᵢ)] − [i died in a tie at exitᵢ]·rᵢ·cᴰ(exitᵢ),
c(t)  = Σ over event times s ≤ t of  w̄·Σᵣ 1/Aᵣ(s),
cᴰ(t) = w̄·Σᵣ (r/k)/Aᵣ(t),      Aᵣ = S₀ − (r/k)·S₀ᴰ,   r = 0, …, k − 1,
```

which is the study's §4.1 written for the grid, and M gains one rank-one term per Efron step. Eᵢ is the row's expected number of events, which the engine already computes for martingale residuals. Probe A-1 confirms these forms against explicit risk sets: Σ Eᵢ equals the weighted number of events, and no Eᵢ is negative [probe A-1].

*Action.* Put both cases in step 4's specification and fixtures. The payload to the driver stays O(K·p).

**F-6. Say which equations define the estimate when providers sit at the bound.** As the study describes it (§4.3), at pprof_py's converged estimate the plain score in β is not zero. What is zero is SerBIN's reduced right-hand side, to which the bounded providers contribute their non-zero scores. The validated estimate is therefore the root of a set of estimating equations, and not the maximizer of the likelihood under bounds. The two differ by 1.4·10⁻⁶ in β, or 2·10⁻⁵ standard errors, in the study's test.

The study's choice, to step through the same reduced system, preserves the validated number, and I agree with it. Three consequences follow.

- The specification of step 5 should state the equations: interior providers solve their own score equation; bounded providers sit at the median plus or minus the bound; β solves the reduced system, including the bounded providers' terms.
- A convergence check has to use the reduced residual. A check built on the likelihood would point at the other root.
- Step control by "halving on a concave function" needs care near convergence, because the root is not a stationary point of the bounded profile likelihood. The second-order difference is far below rounding in the study's test, so I expect no stall, but the specification should say what the acceptance test does there [hypothesis].

*Action.* Record the definition as an explicit statistical choice, and register the 2·10⁻⁵ standard-error distance from the bounded maximum-likelihood estimate, so that the methodology owner can confirm it.

*Implemented, 2026-11-16 (version 0.8).* The `statcompute` engine fits these equations by profile-likelihood Newton, with the providers that have no finite intercept held at the median plus or minus the bound. On data with three such providers it returns pprof_py's coefficients to 4·10⁻¹⁵ standard errors and every intercept, the held ones included, to 4·10⁻¹⁶ [engine tests]. The statistical question is unchanged and still the methodology owner's.

**F-7. A bit-identical interim step exists.** Neumaier's update chooses between two formulas by comparing magnitudes. The branch-free two-sum computes the same rounding error, so the accumulator's state is the same bit for bit: 16 million pairs in four regimes without a mismatch, and identical JVM arrays at p = 20, 50 and 100 [probe A-3, A-4]. It ran 1.2 to 1.3 times faster at p = 50 and 100 [probe A-4]. That is small next to chunking, but it needs no fixture re-comparison, and it applies to the sums that stay compensated: the score, the provider sums and the folds of the chunks.

*Action.* Optional, with step 2. Pin the behaviour at overflow by a unit test: once a sum is not finite the present code ignores the compensation, and the two-sum must end in the same value.

**F-8. A first number for the sparse Gram matrix.** §4.4 leaves the gain from zero covariates unmeasured. In probe A-4 half the columns were indicators. Accumulating over index lists was 2.2 times faster than the chunked dense kernel at a prevalence of 0.30 and about 3 times faster at 0.05, at p = 50 and 100; the limit in that design is 4, because a quarter of the pairs are dense [probe A-4].

Storage may matter more than arithmetic. After vectorization the 1 μs decode is a sixth of an information pass and more than half of a gradient pass [derived from §5.2]. Indicator columns held as index lists would shrink the cached working set and the decode with it [hypothesis].

*Action.* The maintainer states, in aggregate, the share of indicator columns and their prevalence in the production designs (decision 4). Step 0's grid gains one design with that profile.

**F-9. Distinguish a prior-period start from a pilot start.** The study rejects pilot fits, correctly: they are not free, and they end 8 standard errors from the optimum. A production fit has another start available at no cost, the previous period's estimate. In probe A-5 a start 0.01 from the solution, fitted to data that overlapped by 80%, took 2 Newton steps against 3 to 5 from zero [probe A-5].

*Action.* Measure it as an option in step 5's benchmark. If it is adopted, the start is recorded with the fit (NN-11) and the check of F-10 guards the result. With the profile method no provider effects need to be carried over.

**F-10. Turn converged parity into a tolerance class.** The maintainer accepted converged parity for the profile method on 2026-11-13 [study §4.3]. The study also shows that a stopping rule tuned for quadratic convergence stops other methods early (§5.1), which probe A-5 reproduced. Converged parity needs a measure of convergence that does not depend on the path.

*Action.* Define a class, say T-conv: at the reported estimate, the Newton step computed from the exact score (or the reduced residual of F-6) and the exact information is below a stated bound in standard-error units, the study's own unit of deviation. Every fit records the value. It costs nothing where the last pass already returns the score and the information. To first order, two implementations that both meet the bound differ by at most twice the bound. The bound is calibrated against the fixtures, as D-09 calibrated the other classes.

**F-11. Three additions to §4.5's list.**

| Candidate | Evidence | Verdict |
|---|---|---|
| A one-pass variant of §4.2 that weights the Gram matrix by the previous iterate's hazard | 5 to 13 steps against Newton's 3 to 5 [probe A-5] | Rejected |
| Exact information once or twice, then BFGS updates of it | 1 to 2 information evaluations against 4 to 6, in 3 to 9 steps against 3 to 5, for Cox; 2 against 5 for the logistic profile [probe A-5, A-6] | Consistent with the study's verdict on the chord method: not at this width |
| Order-independent accumulators (binned sums, Ahrens et al. 2020†; exact sums, Neal 2015†) in place of fixed reduction groups | Literature only | Not needed if step 1 works; kept as a research note (O-12) |

### 2.4 What the study does not cover

By design it covers two models, the fit only, and pprof_spark only. The rest of this document adds what the brief asks for beyond that: the other families, inference, the literature outside optimization, priorities across the portfolio, and a path to publications.

## 3. Research vision

### 3.1 Thesis and scope

The project studies **estimator-preserving computation**: for a statistical procedure whose definition is fixed, find the cheapest computation that provably returns the same estimate, and say what "the same" means in each case.

It is in scope when a model has all three of these properties:

- a low-dimensional parameter of interest, such as the covariate coefficients;
- one or more high-dimensional blocks that separate, such as provider effects, event-time effects, cluster effects or random effects;
- data too large for one machine's memory, or too slow on one thread.

Cost is a vector, not a number: CPU per row per pass, passes per fit, bytes shuffled, bytes returned to the driver, executor memory, and wall time. The study shows why. On the standard cluster communication is not scarce, so methods designed to save it solve the wrong problem [study §1].

Out of scope: new statistical models; approximate estimators as reported measures; hardware the standard cluster does not have; and a wide-model mode, which the study treats as a separate project [study §5.2].

### 3.2 Equivalence classes

The study's §6 already gives the four classes. This roadmap adopts them and adds the evidence each one requires.

| Class [study §6] | Evidence required | Tolerance |
|---|---|---|
| Exact, same iterates | NUM-2: identical bits, or last bits only with the fixtures re-compared under T-fn and T-iter and a negative control | Existing classes |
| Exact, other iterates | Function-level parity wherever a function is shared; converged parity under T-coef and T-var; the convergence check of F-10 at the reported estimate; iteration counts reported, not compared | T-conv, proposed in F-10 |
| Approximate | A stated distance from the exact estimate, in standard errors, on the benchmark designs; never a reported measure | Not applicable |
| Another estimator | Its own specification and statistical validation (layer T11) | Not applicable |

### 3.3 The structure the case studies share

Both case studies eliminate one separable block exactly, and the result has the same shape.

| | Providers, logistic or linear model | Event times, Cox model |
|---|---|---|
| Eliminated parameter | The intercept γⱼ | The baseline hazard increment at t |
| Group weight | hⱼ = Σ q over provider j | dₜ, the weighted events at t |
| Group mean | x̄ⱼ = bⱼ / hⱼ | x̄(t) = S₁(t) / S₀(t) |
| Weighted Gram matrix | Xᵀ·diag(q)·X | Xᵀ·diag(rᵢcᵢ)·X |
| Reduced information | Xᵀ·diag(q)·X − Σⱼ hⱼ·x̄ⱼx̄ⱼᵀ | Xᵀ·diag(rᵢcᵢ)·X − Σₜ dₜ·x̄(t)x̄(t)ᵀ |
| Solve for the eliminated block | A scalar equation per provider | Closed form: dₜ / S₀(t) |
| Membership of a row | One provider | Every event time in its at-risk interval, hence suffix sums |

The left column is the study's §4.3: the Schur complement, the profile likelihood and within-provider centring are one object. The right column is the study's §4.1, read through the Poisson form of the Cox model, in which each event time has its own effect (Whitehead 1980†) [derived].

Two things follow.

- **One kernel serves every family**: a weighted Gram matrix plus grouped first moments. That is where optimization effort should go, and it is what the study's chunked accumulation already targets.
- **The open problems have two separable blocks.** Providers and event times together give Cox with provider effects. Providers and clusters together give the three-stage readmission model. One block can still be eliminated exactly; the other then leaves a matrix that is diagonal minus low rank, or sparse. Section 4 takes these up.

### 3.4 Contributions the project can aim for

| Contribution | Kind | How new it is likely to be |
|---|---|---|
| Exact Cox regression on distributed data, on gridded times, with no sort and no second moments per time, including Efron ties and entry times | Method and software | A new combination, subject to the search of F-3 |
| Exact Newton steps for the Cox model with provider effects (O-7) | Method | Possibly new; untested |
| Exact provider limits from one recursion (O-5) | Method, small | The identity is classical; its use here appears to be new to these packages |
| Statistical fitting that is bitwise invariant to partitioning and cluster size, with the evidence regime that supports it | Software and practice | Uncommon in statistical software; worth describing in its own right |
| Shared kernels: weighted Gram, grouped moments, time-grid aggregates, block elimination | Software | Engineering |
| A documented way of doing computational research with AI assistants: evidence classes, rejected candidates kept, independent review | Practice | Timely; depends on keeping the record honest |

Most of the individual identities are old. The study says so for the logistic model, and it is true of the Cox identity as well. The value lies in combining them correctly, proving equivalence, and measuring.

### 3.5 Working with AI as a research partner

The study is the model to follow: every statement tagged by its evidence, rejected candidates kept with their numbers, limits stated. Three additions make the practice a research method.

- **A register of hypotheses**, each with its class, its evidence and the measurement that would reject it (section 8.3).
- **Independent review as routine.** This review derived the Cox identity separately and tested it on different data. That is how F-4 was found: the study's designs did not spread stratum means, so they could not show the cancellation.
- **An automated evaluator.** AlphaEvolve's results rest on running every machine-proposed program through an automated evaluator (Google DeepMind 2025†). This project already has one: the fixtures, function-level parity and the kernel benchmarks. Candidate kernels can be generated freely as long as that evaluator decides.

The limits stay as the project context states them: assistants see no protected data and do not touch the cluster, and evidence is labelled with the environment that produced it.

## 4. Research opportunities

### 4.1 Overview

| ID | Opportunity | Structure used | Class | Evidence today |
|---|---|---|---|---|
| O-1 | Chunked accumulation and reduction on the executors | Sums over rows; fixed reduction groups | Exact, same iterates | Measured [study §4.4] |
| O-2 | Weighted-Gram information for Cox | Poisson form; expected events as weights | Exact, same iterates | Measured for Breslow without entries [study §4.1]; identity checked for the rest [probe A-1] |
| O-3 | Per-time aggregates for unstratified fits and large strata | Gridded times; first moments only | Exact | Measured on 60,000 rows [study §4.2] |
| O-4 | Profile-likelihood Newton for the logistic model | Scalar solves inside the block | Exact, other iterates | Measured [study §4.3]; [probe A-6] |
| O-5 | Exact provider limits by exponential tilting | The event count is a sufficient statistic | Exact | [probe A-2] |
| O-6 | Indicator-aware block layout and Gram matrix | Zeros in the design | Exact, last bits | First number [probe A-4] |
| O-7 | Exact Newton for Cox with provider effects | Two diagonal blocks, providers and event times | Exact, other iterates | Algebra only [hypothesis] |
| O-8 | Fewer passes in random-effect fits | Several parameter points per pass | Exact, other iterates | About 200 jobs per fit today [repo, OI-99] |
| O-9 | Crossed provider and cluster terms | Sparse factorization; selected inversion; backfitting | Exact, or iterative | Literature |
| O-10 | Covariate likelihood-ratio tests without a full refit each | Restricted one-step starts; several fits per pass | Exact, other iterates | [hypothesis] |
| O-11 | Residuals and robust variance from the time grid | The same grid tables as O-3 | Exact | [derived]; not prototyped |
| O-12 | Order-independent accumulators | Binned or exact summation | Exact | Literature † |
| O-13 | A wide-model mode | Truncated Newton on profiled, standardized objectives | Exact, other iterates | Crossover at 400 to 1,000 features [study §5.2] |

O-1 to O-4 are the study's, and section 2.3 holds what I would add to them. The rest of this section describes the new ones in the form the brief asks for.

### 4.2 O-5: exact provider limits by exponential tilting

**Why the present approach is slow.** The exact interval inverts the test by root finding. Every trial value g of the provider effect re-runs the Poisson-binomial recursion, at O(nⱼ·min(oⱼ, nⱼ − oⱼ)) each, 20 to 40 times per limit [repo, provider-tests specification].

**The structure.** With logit πᵢ(g) = g + aᵢ, the count S of events is the sufficient statistic of a one-parameter exponential family:

```text
P_g(S = s) = P_g₀(S = s) · exp((g − g₀)·s) / M(g),
M(g) = Πᵢ [1 − πᵢ(g₀) + πᵢ(g₀)·exp(g − g₀)].
```

**How it works.** Run the recursion once at g₀ = γ̂ⱼ, where the stored probabilities are well scaled around the observed count. Each trial value then costs O(oⱼ) for the tilted sum, in log space, and O(nⱼ) for log M through log1p and expm1. Keeping a second recursion over the non-events gives the other tail with full relative accuracy by the same formula, with the sign of the shift reversed.

**Estimator.** Unchanged. The probabilities are the same up to rounding, so the comparison is under T-test and T-p.

**Cost.** From (evaluations) × O(nⱼ·oⱼ) to two recursions plus (evaluations) × O(nⱼ). Memory is O(nⱼ) per provider.

**Risks.**

- The reference clips probabilities to [10⁻¹⁰, 1 − 10⁻¹⁰] at every trial value. Tilting assumes the logistic form exactly, so a row at the clip breaks the identity. Rule: fall back to the direct recursion for a provider with any clipped row.
- Stored probabilities far from the observed count underflow. They are negligible there, but the code must not take their logarithm blindly.
- Binomial rows expand into trials, as now, up to the same cap.

**Evidence.** On 4,000 rows with 958 events, 25 shifts whose tails ran from 1 down to 10⁻⁵⁵ agreed with the direct recursion to 6.6·10⁻¹⁴. Forty evaluations took 0.012 s against 0.440 s in NumPy [probe A-2].

**Validation and benchmark.** Compare with `PoissonBinomial.tails` at every trial value the existing fixtures visit, and compare the limits to the root finder's tolerance; keep the specification's negative controls. Benchmark the test pass at a point with skewed provider sizes, where OI-57 expects stragglers.

### 4.3 O-6: an indicator-aware block layout

**Why.** Blocks hold every covariate as a double in row-major order [repo]. Risk-adjustment designs often contain many indicators. Dense storage then spends memory, decode time and arithmetic on zeros.

**The structure.** A row with z non-zero covariates costs z²/2 instead of p²/2 in the Gram term, exactly, provided only the dense columns are centred [study §4.4]. Indicators can be stored as index lists.

**Estimator.** Unchanged; the last bits move. Centring of the indicator columns moves into the rank-one terms, and that correction needs its own error analysis.

**Evidence.** 2.2 to 3 times over the chunked dense kernel with half the columns as indicators [probe A-4]. No measurement of memory or decode yet.

**Risks.** More complex kernels and a second block format; rare indicators and conditioning; a gain that depends entirely on the designs.

**First test.** The design facts of decision 4, then one indicator-heavy design in step 0's grid.

### 4.4 O-7: exact Newton for the Cox model with provider effects

**The present approach.** pprof_py's provider-effect Cox model has one shared baseline hazard and an effect γⱼ per provider. It alternates: one Newton step per provider on γ using the diagonal bound "information of provider j ≤ its expected events", then a solve for β [repo, `algorithms/survival/provider_effects.py`]. That is an alternating scheme of BAN's kind. The study records what BAN costs in the logistic model once the blocks are coupled: 19 iterations, then 705, then no convergence in 10,000 as covariates shift [study §3].

**The structure.** In the Poisson form the model has two diagonal blocks, providers and event times, with a dense cross block between them and the covariates bordering both. Eliminating the event times gives the Cox information for (γ, β):

```text
I(γ, β) = A − Σₜ dₜ·v(t)v(t)ᵀ,      v(t) = (π(t), x̄(t)),
A      = the arrowhead matrix with diagonal Eⱼ, border Σ over provider j of Eᵢxᵢ, and corner Xᵀ·diag(Eᵢ)·X,
πⱼ(t)  = provider j's share of the risk-set weight at t.
```

So the provider block is diagonal minus a term of rank at most K, the number of event times. Eliminating the providers instead leaves a dense system of order K + p, about 2,000 for daily times over five years and 180 features [derived, from the identity of F-5 with provider indicators as covariates].

**How it would work.** Two passes per evaluation, as in the per-time layout. Pass 1 returns S₀(t) and S₁(t). Pass 2 forms the weighted Gram matrix and, per provider, its K risk sums, and accumulates Σⱼ pⱼpⱼᵀ / Eⱼ on the executors as a packed K × K partial, about 13 MB at K = 1,825. The driver solves the system of order K + p. One linear constraint removes the direction in which all γ shift and the baseline absorbs it.

**Estimator.** The same partial-likelihood maximizer. Iterates change. The median bound raises the question of F-6 again.

**Cost.** Per evaluation, the usual O(n·p²), plus O(m·K²/2) for the provider cross-products: 1.3·10¹⁰ multiply-adds at 8,000 providers. That grows to 1.7·10¹² at a million providers, where conjugate gradients on the same system, preconditioned by the present diagonal step, would be the alternative [derived].

**Risks.** K near 10⁴ makes the K × K partials too large; Efron ties add a rank-one term per step; the gain is unknown until the present method's iteration counts are measured.

**Evidence.** None beyond the algebra.

**First test.** A NumPy probe against pprof_py's `ProviderPenalizedCoxPH` at λ = 0: the same optimum to converged parity, and the passes each needs, on the study's two designs and on one with provider-level case-mix differences.

### 4.5 O-8: fewer passes in random-effect fits

**Observation.** A logistic random-intercept fit runs about 200 Spark jobs: a grid and a Brent search over σ with Newton iterations at each value, then a bounded quasi-Newton search with the analytic gradient [repo, OI-99 and the random-effect estimation specification]. Each evaluation is one pass.

**Ideas [hypothesis].**

- Evaluate the whole σ grid in one pass per Newton sweep. The eight-step line-search ladder already evaluates eight points per pass.
- Start the Laplace search from a moment estimate of σ, computed from a fixed-effect fit's γ̂ⱼ and their variances, where the first stage serves only as a start.
- Use the analytic Hessian of the Laplace deviance, so the final search is Newton's method.

**Estimator.** The same objective and optimum; other iterates.

**First test.** Count evaluations on the existing random-effect fixtures with the in-memory kernel. No cluster is needed.

### 4.6 O-9 to O-13 in brief

- **O-9, crossed terms.** Stage 2 of the three-stage model factorizes a dense Schur complement on the smaller set of levels [repo, OI-59]. A sparse factorization with a fill-reducing ordering, selected inversion for the effects' variances, and backfitting (Ghosh, Hastie and Owen 2022†) are the candidates. Wait for the measurements at the scale plan's points A10 and A11.
- **O-10, covariate tests.** Each restricted fit can start from the one-step restricted estimate, β̃ = β̂ − S⁻¹eₖ·β̂ₖ / (S⁻¹)ₖₖ, and the restricted fits can share passes. Today each is a full refit after a rebuild [repo].
- **O-11, row-level outputs from the grid.** In the per-time layout a row's score residual needs c(t) and the cumulative integral of x̄ dΛ at its entry and exit times. Both are tables from pass 1, so residuals and the robust variance need no sort either [derived].
- **O-12, order-independent sums.** Binned accumulators give the same sum in any order, and exact accumulators give the correctly rounded sum (Ahrens et al. 2020†; Neal 2015†). Reductions could then run in any order on the executors. This competes with the fixed reduction groups of the study's step 1, which are simpler.
- **O-13, wide models.** As the study's step 6.

### 4.7 Rejected or withdrawn

| Candidate | Evidence | Verdict |
|---|---|---|
| BAN, L-BFGS, Newton-CG, the chord method, the communication-saving optimizers | [study §5] | Rejected at about 180 features |
| Pilot starts; approximate estimators, as answers or as starts | [study §5.1, §6] | Rejected for reported measures |
| Information reused with BFGS updates | [probe A-5, A-6]; saves a fifth to a quarter of CPU for more rounds and a new stopping rule [study §5.2] | Not at this width |
| Lagged-hazard one-pass Cox evaluation | 5 to 13 steps against 3 to 5 [probe A-5] | Rejected |
| GPU kernels | Not on the standard cluster, and kernels are not yet shown to be the wall-clock term | Out of scope |

## 5. Literature landscape

The study's reference list covers optimizers, distributed estimation and the fixed-effect literature, and it marks the entries it checked. This section adds the areas it does not cover. Appendix B gives the full entries.

| Area | Reference | What it gives | Use here |
|---|---|---|---|
| Cox as a Poisson model | Whitehead (1980)† | An auxiliary Poisson model whose likelihood is proportional to the partial likelihood | The origin of the weighted-Gram identity; a way to derive algorithms for two-way structures (O-7) |
| Distributed Cox on a time grid | H2O-3 `CoxPH` (source read) | Grid aggregates on the workers, with second moments per time | The closest prior art for the per-time layout (F-3) |
| Scans for massive samples | Yang, Schuemie, Ji and Suchard (2024)† | Single-pass scans for Cox and forward-backward scans for Fine–Gray, on GPUs, within cyclic coordinate descent | Prior art for scans; the route into Fine–Gray |
| | Kawaguchi, Shen, Suchard and Li (2021)† | A forward-backward scan that takes Fine–Gray from O(n²) to O(n), shown on renal registry data | The next survival family after Cox |
| Structure in Cox optimization | Liu, Zhang and Rudin (2024)† | Exact coordinate-wise derivatives in linear time, and surrogate functions with monotone descent | Penalized paths; an example of the kind of result this project looks for |
| Fixed effects as nuisance parameters | Stammann (2018)† | A weighted Frisch–Waugh–Lovell step with alternating projections inside Newton–Raphson, for k-way effects | The same algebra as SerBIN for one factor; a candidate for crossed terms (O-9) |
| | Fernández-Val and Weidner (2016)† | Bias corrections for nonlinear models with fixed effects | "Another estimator" in the study's classes; relevant only if the bias of small providers' effects is ever in question |
| Exact conditional inference | Hirji, Mehta and Patel (1987)† | A recursion for the exact distributions of sufficient statistics in logistic regression | The machinery behind O-5 |
| Reproducible summation | Ahrens, Demmel and Nguyen (2020)†; Demmel and Nguyen (2015)† | A binned accumulator whose result does not depend on the order of summation | O-12 |
| | Neal (2015)† | Exact summation at less than twice the cost of simple summation on large arrays | O-12 |
| Crossed random effects | Ghosh, Hastie and Owen (2022a)† | Backfitting that computes the generalized least-squares estimate at O(N) cost under stated conditions | O-9, linear models |
| | Ghosh, Hastie and Owen (2022b)† | The logistic extension, by penalized quasi-likelihood | O-9, but a different estimator from the Laplace fit |
| Systems | McSherry, Isard and Murray (2015)† | COST: the configuration at which a parallel system beats one competent thread | The framing of step 0 and of PERF-1 |
| | Wong, Forsell, Lewis, Mao and Wardrop (2021)† | Lossless compression of data to conditionally sufficient statistics for linear models | The idea behind binomial compression [study §4.4], for narrow models |
| Large-sample approximations | Keret and Gorfine (2023)†; Wang et al. (2021)† | Optimal subsampling that keeps every failure; divide and conquer with one-step updates | The study's "approximate" class; not for reported measures |
| Machine-proposed algorithms | Google DeepMind (2025)† | AlphaEvolve: candidate programs kept or discarded by an automated evaluator | Section 3.5 |

Five ideas look under-used in this field.

- **The Poisson form as a tool for deriving algorithms**, and not as a way to fit a Cox model with GLM software.
- **Exponential tilting for exact limits.** Exact conditional inference has used the underlying recursion for decades.
- **Exact invariances as numerical tools.** A stratum-specific shift costs nothing statistically and removes a cancellation (F-4).
- **A competent single-thread baseline.** The study already measures one, pprof_py at 17 and 2.3 μs per row.
- **Bitwise reproducibility as a requirement**, with the summation literature that makes it affordable.

Two areas I cite from memory and did not check: selected inversion of sparse factors, for the variances in O-9, and learning over joins from the database literature, which is the general form of the weighted-Gram idea for person-period data.

## 6. The two case studies as starting projects

| Question | Cox | Logistic fixed effects |
|---|---|---|
| Is structure left to exploit? | Yes: the Gram identity and the per-time layout [study §4.1, §4.2] | No: the Schur complement, the profile likelihood and centring are one object [study §4.3] |
| Derived gain in CPU per fit | 312 to 38 μs per row, about 8 times [study §5.2] | 237 to 60 μs per row, then 39 with the profile method [study §5.2] |
| Structural limit removed | One task per pass for unstratified fits and large strata | Two passes per step, and γ shipped on each |
| Research novelty | Moderate, subject to F-3 | Low for the fit; some in inference (O-5) |
| Validation assets | Fixtures with pprof_py and R references; three parity levels [repo] | Fixtures with pprof_py and R pprof references; three parity levels [repo] |
| Largest open risk | Wall-clock time (F-1); cancellation without per-stratum centring (F-4) | The definition of the estimate at the bound (F-6) |
| What it gives the portfolio | The time-grid layout and the shared kernel | The convergence class (F-10) and the inference kernels |

They are the right first projects. Both have validated references and measured gains, and both force the infrastructure every later family needs: the kernel, the reduction and the convergence class. They are not where most of the research novelty is. Plan them as the foundation, and take the research questions from section 4.

What stays open in each after the study:

- **Cox.** Engine-level parity of the Gram form with Efron ties and entry times (step 3, with F-4 and F-5); the per-time layout with several large strata, where the grid tables multiply by the number of strata; continuous times, where the study keeps sorted scans in three passes; row-level outputs from the grid (O-11); provider effects (O-7).
- **Logistic.** The profile method with bounds, weights and trials (step 5, with F-6); exact limits (O-5); covariate tests (O-10); the median of a million provider effects without holding them on the driver.

## 7. Prioritization

Scores run from 1 to 5, with 5 best: mathematical interest (M), practical impact (P), feasibility (F) and reuse across families (R). The rank is a judgement and not the sum, because an item that gates others ranks above its score.

| Rank | Item | M | P | F | R | Evidence today | Reason for the rank |
|---:|---|:-:|:-:|:-:|:-:|---|---|
| 1 | Step 0 with wall-clock attribution (F-1, F-2) | 1 | 5 | 5 | 5 | Kernel rates only [study] | It gates every other choice |
| 2 | O-1: chunked accumulation and executor-side reduction | 2 | 5 | 5 | 5 | Measured [study] | The largest measured CPU term, shared by every family |
| 3 | O-2 and O-3: weighted-Gram Cox and the per-time layout, with F-4 and F-5 | 4 | 5 | 4 | 4 | Measured [study]; [probe A-1, A-7] | Removes the one-task limit; the one methods paper within the case studies |
| 4 | O-4: profile-likelihood Newton, with F-6 and F-10 | 3 | 4 | 4 | 3 | Measured [study]; [probe A-6] | Half the rounds; the convergence class it needs serves every later method |
| 5 | O-5: exact limits by tilting | 3 | 3 | 5 | 3 | [probe A-2] | Small and self-contained; needs no cluster |
| 6 | O-6: indicator-aware layout | 2 | 3 to 5 | 3 | 5 | [probe A-4] | Its impact depends on the designs (decision 4) |
| 7 | O-7: Cox with provider effects | 5 | 3 | 3 | 3 | Algebra | The best research candidate |
| 8 | O-8: passes of random-effect fits | 3 | 4 | 3 | 3 | Job count [repo] | About 200 jobs per fit today |
| 9 | O-10: covariate tests | 2 | 3 | 4 | 3 | None | One refit per covariate today |
| 10 | O-9: crossed terms | 4 | 3 | 2 | 3 | Literature | Wait for the measurements at A10 and A11 |
| 11 | F-9: prior-period start | 1 | 2 | 5 | 4 | [probe A-5] | Free, if the governance of results allows it |
| 12 | O-12: order-independent sums | 4 | 2 | 3 | 4 | Literature | Only if step 1 disappoints |
| 13 | O-13: wide-model mode | 3 | 2 | 3 | 3 | [study §5.2] | A separate project |

A note after the maintainer's answers of 2026-11-14: the recorded Cox fits were provider-stratified, so the per-time layout (O-3) would not have touched them. O-3 keeps its place for unstratified fits and as the candidate methods contribution. For provider-stratified production fits the gain of rank 3 comes from O-2 alone, and rank 1 matters more.

## 8. Development roadmap

### 8.1 Four tracks

| Track | Content | Depends on the cluster |
|---|---|---|
| A. Case studies | The study's steps 0 to 6, in its order, with the amendments of section 2.3 | Yes, for every exit measurement |
| B. Exact inference | O-5, then O-10, then the robust-variance aggregation of OI-45 | No for the kernels; yes for the benchmarks |
| C. Second-wave structures | O-7, O-8, O-9 | No for probes and specifications |
| D. Research record and publications | The register (8.4), the probes, the papers (8.5) | No |

### 8.2 Track A: the study's steps with the amendments attached

| Step [study §8] | Amendments | Exit evidence added to the study's |
|---|---|---|
| 0. Baseline on current main | F-1, F-2, F-8 | F-1's three preconditions met first. Effective parallelism and straggler ratio per pass; shares of wall time by phase; totals per job with every output; one indicator-heavy design |
| 1. Reduction groups with plan-set partitioning | None | As the study |
| 2. Chunked accumulation | F-7 | A bit-identity test for the two-sum, including the overflow case |
| 3. Weighted-Gram form with Efron ties and entry times | F-3, F-4, F-5 | A fixture with stratum means spread by 100 within-stratum SDs, inside T-fn; a negative control |
| 4. Per-time aggregates | F-3, F-5, O-11 | Efron and entry-time fixtures in the grid layout |
| 5. Profile-likelihood Newton | F-6, F-9, F-10 | The convergence value recorded with every fit; the prior-period start measured |
| 6. Wide-model decision | None | As the study |

### 8.3 Order and gates

Work that needs no cluster can start now, in parallel with step 0:

- the tilting kernel of O-5 in `numerics`, checked against the existing tails;
- the NumPy probe of O-7 against pprof_py;
- the evaluation counts of O-8 on the in-memory kernel;
- the specifications for F-4, F-5 and F-6;
- the prior-art search of F-3.

| Gate | When | Passes if |
|---|---|---|
| G0 | After step 0 | Wall time is attributed to phases, and the low effective parallelism of the recorded runs is explained or no longer appears. The order of steps 1 to 5 is confirmed or changed |
| G1 | After steps 1 and 2 | CPU per row per pass on the cluster is near the study's JVM rates; results are bitwise equal across partition counts |
| G2 | After steps 3 to 5 | Parity at the levels each class requires; an unstratified fit at 20 million rows uses every worker |
| G3 | After the probes of tracks B and C | The first second-wave project is chosen on evidence (decision 6) |

I give no dates. The repository moves in rounds, and decision D-50 already orders its milestones; these tracks fit beside them.

### 8.4 The research register

One table, kept with the project's other state files, with one row per hypothesis.

| Field | Content |
|---|---|
| ID and statement | For example "O-7: exact Newton for provider-effect Cox converges in a single-digit number of evaluations" |
| Class | One of the four of section 3.2 |
| Evidence | Tags as in this document, with the probe or run that produced each |
| Rejecting measurement | Stated before the test. For O-7: "no fewer evaluations than the alternating scheme on the three probe designs" |
| Status | Proposed, probed, specified, adopted, rejected or withdrawn |
| Decision | The decision number, once the maintainer has ruled |

Rejected rows stay. The study's §4.5 and section 4.7 here are its first entries.

### 8.5 Publications

| Candidate | Depends on | Main risk |
|---|---|---|
| P1. Exact Cox regression on distributed data without a sort or second moments per time | Steps 3 and 4; the search of F-3; cluster results | Prior art found by the search |
| P2. A software paper on pprof_spark: the methods, the parity levels and bitwise partition invariance | The scale test; a release | Scope: it needs measured scale results |
| P3. Exact Newton for the Cox model with provider effects | O-7 succeeding | The gain may be small on real designs |
| P4. A case study of computational research with AI assistants: evidence classes, independent review, rejected candidates | The register kept honestly over several projects | It must report failures as fully as successes |

Likely venues are the computational statistics and statistical software journals for P1 to P3, and a methodology or data-science journal for P4. I have not checked any journal's current scope.

## 9. Decisions requested

1. **Track A.** Adopt the study's steps 0 to 6 as track A, with amendments F-1 to F-11, or say which amendments to drop.
2. **The estimate at the bound (F-6).** Confirm that the validated definition is the root of SerBIN's reduced system, and whether to register its distance from the bounded maximum-likelihood estimate.
3. **A convergence class (F-10).** Add T-conv to §8.4 of the project context, with a bound calibrated on the fixtures.
4. **Design facts, in aggregate only.** The share of indicator columns and their typical prevalence; the share of rows with entry times; the number of distinct event times per stratum; the size of the largest stratum and provider; which outputs a production job requests.
5. **The benchmark environment.** Confirmed by the maintainer on 2026-11-14: the cluster has 10 workers of 16 GB (it used to have 30); the target runtime is 17.3 LTS. Still needed: that the cluster is of fixed size, and whether pprof_spark, whose JAR targets Spark 4.1, is to be benchmarked as built or rebuilt for Spark 4.0 by the maintainer.
6. **The first second-wave project.** O-7, O-8 or the inference track, once their probes report.
7. **Where the research record lives.** Answered on 2026-11-14: in a repository of its own. Proposed name `statcompute`; the register, the probes and this document are in its `docs/research/` and `statcompute/lab/probes/`.

## Appendix A. Sandbox probes

Environment: Python 3.12.3 with NumPy 2.5.4, and OpenJDK 21.0.12, on one core of an Intel Xeon at 2.3 GHz. Seeds are fixed, so every figure except the timings reproduces exactly. The complete output is in `roadmap-evidence.md`; `run_all.sh` in `roadmap-probes.tar.gz` regenerates it.

| ID | Question | Setup | Result |
|---|---|---|---|
| A-1 | Does the weighted-Gram form equal the classical information with Efron ties, entry times, weights, offsets and strata? | 600 rows, 4 covariates, 5 strata, integer times from 1 to 24, 40% of rows with entry times, weights 1 to 3; compared with explicit risk sets and with central differences of the score | Largest difference over largest entry: 1.2·10⁻¹⁵ (Breslow), 7.8·10⁻¹⁶ (Efron); against central differences 8·10⁻¹² and 1.3·10⁻¹¹; scores identical; Σ Eᵢ = 739 = the weighted events; no negative Eᵢ |
| A-2 | Do tilted Poisson-binomial tails equal recomputed ones? | 4,000 rows, 958 events, 25 shifts of ±0.6 around the provider's own effect | Worst relative difference 6.6·10⁻¹⁴, over tails from 1 to 10⁻⁵⁵; 40 evaluations in 0.012 s against 0.440 s |
| A-3 | Is the two-sum's error term Neumaier's, bit for bit? | 16 million pairs: wide exponents, similar magnitudes, near cancellation, subnormals and zeros; a running sum of 300,000 terms | No mismatch; identical final state |
| A-4 | What do chunking and index lists gain in a JVM Gram kernel? | 20,000 rows; half the columns continuous, half indicators at prevalence 0.30 or 0.05; best of 7 runs after warm-up; p = 50 and 100 | Per-element Neumaier: 1.2 to 1.4 ns per element. Two-sum: 1.2 to 1.3 times faster, bits identical. Chunks of 64 rows: 2.4 to 2.5 times, largest relative difference 1.6·10⁻¹⁶. Index lists: 5.3 to 5.5 times at 0.30 and 7.3 to 7.6 times at 0.05 |
| A-5 | How do Cox iterations behave when the score is exact and the information is reused? | 200,000 rows, daily times, 20 and 60 covariates, three designs; every method stopped by the same Newton-decrement rule at 10⁻¹⁰, except the first row | Under the relative-ℓ rule, information frozen after one step stopped at a largest score of 0.26, 2·10⁻⁵ from the root. Under the common rule: Newton 3 to 5 steps; lagged hazard 5 to 13; information once or twice then BFGS 3 to 9 steps with 1 to 2 information evaluations; prior-period start 2 steps |
| A-6 | Does profile-likelihood Newton reach the joint optimum in fewer passes? | 207,330 rows, 1,994 providers, 20 covariates, no provider at the bound | Joint Newton by block elimination: 6 steps, 13 passes at two per step. Profile Newton: 4 steps, 5 passes, β within 2·10⁻¹². Two Schur complements then BFGS: 5 steps, 6 passes |
| A-7 | How does the spread of stratum means affect each form of the Cox information? | 3,000 rows, 30 strata, Breslow ties; one covariate's stratum means spread by 0 to 1,000 within-stratum SDs | The table of F-4 |

Limits of the probes:

- They ran on toy data, on one core. None ran against the engine or its fixtures, and none is evidence about wall time on a cluster.
- The NumPy sums are not the engine's compensated sums, so the accuracy figures of A-1 and A-7 are upper bounds on what the engine would show, and the cancellation of A-7 is not.
- A-4 is not a JMH benchmark. Its runs at p = 20 last 5 ms and are left out of the figures above.
- A-5 and A-6 use one design family each. Their pass counts indicate direction, as the study's own counts do.

## Appendix B. References added by this document

A dagger marks an entry checked on 2026-11-13 against the record named in brackets. The study's own reference list is not repeated.

† Ahrens P, Demmel J, Nguyen HD (2020). Algorithms for efficient reproducible floating point summation. ACM Transactions on Mathematical Software 46(3), Article 22. doi:10.1145/3389360. [publisher record]

† Demmel J, Nguyen HD (2015). Parallel reproducible summation. IEEE Transactions on Computers 64(7):2060–2070. [reference list of the entry above]

† Fernández-Val I, Weidner M (2016). Individual and time effects in nonlinear panel models with large N, T. Journal of Econometrics 192(1):291–312. doi:10.1016/j.jeconom.2015.12.014. [the DOI, as cited on CRAN's page for the alpaca package; title and pages from memory]

† Ghosh S, Hastie T, Owen AB (2022a). Backfitting for large scale crossed random effects regressions. Annals of Statistics 50(1):560–583. doi:10.1214/21-AOS2121. [published article]

† Ghosh S, Hastie T, Owen AB (2022b). Scalable logistic regression with crossed random effects. Electronic Journal of Statistics 16:4604–4635. doi:10.1214/22-EJS2047. [published article]

† Google DeepMind (2025). AlphaEvolve: a coding agent for scientific and algorithmic discovery. White paper, May 2025. [title and method as reported in secondary sources; the paper itself was not read]

† H2O.ai. h2o-3, `h2o-algos/src/main/java/hex/coxph/CoxPH.java`, master branch. [source read on 2026-11-13]

† Hirji KF, Mehta CR, Patel NR (1987). Computing distributions for exact logistic regression. Journal of the American Statistical Association 82(400):1110–1117. [index record]

† Kawaguchi ES, Shen JI, Suchard MA, Li G (2021). Scalable algorithms for large competing risks data. Journal of Computational and Graphical Statistics 30(3):685–693. doi:10.1080/10618600.2020.1841650. [PubMed Central record]

† Keret N, Gorfine M (2023). Analyzing big EHR data: optimal Cox regression subsampling procedure with rare events. Journal of the American Statistical Association 118(544):2262–2275. [index record; also checked by the study]

† Liu J, Zhang R, Rudin C (2024). FastSurvival: hidden computational blessings in training Cox proportional hazards models. Advances in Neural Information Processing Systems 37. arXiv:2410.19081. [arXiv and conference records]

† McSherry F, Isard M, Murray DG (2015). Scalability! But at what COST? 15th Workshop on Hot Topics in Operating Systems (HotOS XV). USENIX. [USENIX record]

† Neal RM (2015). Fast exact summation using small and large superaccumulators. arXiv:1505.05571. [arXiv record]

† Stammann A (2018). Fast and feasible estimation of generalized linear models with high-dimensional k-way fixed effects. arXiv:1707.01815. [arXiv record; also in the study's list]

† Wang Y, Hong C, Palmer N, Di Q, Schwartz J, Kohane I, Cai T (2021). A fast divide-and-conquer sparse Cox regression. Biostatistics 22(2):381–401. arXiv:1804.00735. [arXiv record; journal details as the study gives them]

† Whitehead J (1980). Fitting Cox's regression model to survival data using GLIM. Applied Statistics 29(3):268–275. doi:10.2307/2346901. [index record]

† Wong J, Forsell E, Lewis R, Mao T, Wardrop M (2021). You only compress once: optimal data compression for estimating linear models. arXiv:2102.11297. [arXiv record]

† Yang J, Schuemie MJ, Ji X, Suchard MA (2024). Massive parallelization of massive sample-size survival analysis. Journal of Computational and Graphical Statistics 33(1):289–302. doi:10.1080/10618600.2023.2213279. [index record]

Not checked, cited from memory: Laird N, Olivier D (1981). Covariance analysis of censored survival data using log-linear analysis techniques. Journal of the American Statistical Association 76:231–240.

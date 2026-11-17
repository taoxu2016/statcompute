# Prior art for the engine's Cox computations

Search of 2026-11-17, for OPEN_ITEMS OI-22 and the roadmap's amendment F-3. The question: is anything in
the engine's Cox regression new, in particular the time-grid layout for fits without strata
(`docs/spec/engine.md`, section 2.2)?

## 1. What was claimed, piece by piece

| Piece | Where in the engine |
|---|---|
| A. The information as X' diag(E) X minus one rank-one term per event time, E being each row's expected events | Both layouts |
| B. The same with Efron ties, entry times, case weights and strata | Both layouts |
| C. Per distinct time, aggregates of first moments only (p + 1 numbers), added over partitions, then suffix sums on the driver | Time grid, pass 1 |
| D. A second pass that forms the weighted Gram matrix, so that no second moment per time is ever aggregated | Time grid, pass 2 |
| E. Efron's steps on the grid through three weights per time | Time grid, driver |

## 2. What was found

| Source | What it does | Bears on |
|---|---|---|
| Whitehead (1980); Carstensen, "Who needs the Cox model anyway?"; the glum documentation's Cox tutorial | The Cox partial likelihood is a Poisson likelihood with one effect per event time profiled out. GLM software uses it by expanding the data to one row per subject and period | A: the identity is the information of that Poisson model. Not new |
| Breheny, lecture notes on the Cox score and Hessian (University of Iowa, 2015) | Writes the Hessian as -X'WX with W = diag(Pd) - P diag(d) P', P holding each subject's share of each risk set. Pd is the vector of expected events | A, stated outright in teaching material |
| **coxdev** (Taylor, Hastie, Narasimhan; R and Python packages; R version 0.4.0 of 2026-11-07) and the glmnet vignette "Some notes on the score and Hessian of weighted Cox's proportional hazards with ties" (Taylor, Hastie, Narasimhan, Tay) | The deviance, its gradient and the information as a matrix-free operator, each product costing O(n) through cumulative sums over sorted data; Breslow and Efron ties, start times as a "difference of cumsums" over stop order and start order, strata, weights. Its manual gives the coefficients' information as crossprod(X, I %*% X) | A and B. This is the same algebra as the engine's kernels, published, by the glmnet authors. In memory, sorted, one machine |
| WebDISCO (Lu et al. 2015, JAMIA) | Exact distributed Newton-Raphson across sites. Each site sends, for every distinct event time and every iteration, the risk-set sums of exp(z'b), z exp(z'b) and zz' exp(z'b) | C with second moments. Andreux et al. (2020) put its cost at O(TP) numbers per site for the gradient and O(TP^2) more for the Hessian |
| H2O-3 `CoxPH` (source read 2026-11-13) | Workers aggregate risk-set sums on a time grid, with a time x p x p array per stratum; the driver takes reverse cumulative sums | C with second moments, on a cluster |
| Her et al., distributed regression with SAS, part II (arXiv 1808.02392) | Distributed Newton-Raphson with Breslow ties; without strata the partners' sums are per event time; with one stratum per partner only a p x p Hessian contribution travels | C with second moments; and the observation that strata remove the problem |
| FedECA (Ogier du Terrail et al. 2023) | Builds its pooled-equivalent second-order method on WebDISCO | No change to the aggregates was found in what was read |
| coxstream (PyPI) | Exact Efron estimate out of core: one time-sorted pass per Newton iteration, O(p^2) state | Another route to exact large-n fits. Needs the data sorted by time; one machine |
| ODAC (Duan et al. 2020), COLSA (Hu, Shi and Song 2024), discrete-time federated Cox (Andreux et al. 2020) | Surrogate likelihoods, other likelihoods or other models, to avoid iterating over risk-set aggregates | They change the estimator. Not comparable with an exact fit |

## 3. Verdict

- **A and B are not new.** The identity is old, and coxdev publishes it in operator form with every
  option the engine has. A paper cannot present the weighted-Gram form, or its Efron and entry-time
  versions, as a contribution. The engine's kernels are an independent implementation of known
  algebra, and must cite coxdev and the glmnet vignette.
- **C is not new.** Aggregating risk-set quantities per distinct event time across workers or sites
  is what WebDISCO and H2O-3 do.
- **D was not found.** Every exact distributed method found sends second moments per event time:
  O(Kp^2) numbers per worker. Replacing them by first moments per time and one more pass that returns
  a single p x p matrix, O(Kp + p^2) in all, was not described in any source read. In federated terms
  it would cut WebDISCO's bandwidth by a factor of about p at the price of one more round per
  iteration. It follows in a few lines from coxdev's operator, so an expert may call it immediate.
- **E was not found** and is small: a consequence of doing D on a grid.

So the honest description is: no new estimator and no new identity; one arrangement of known pieces
for distributed data that the sources checked do not contain, with a measurable saving in what
travels to the driver.

## 4. Consequences for the publication plan

| Roadmap item | Before | Now |
|---|---|---|
| P1, "Exact Cox regression on distributed data without a sort or second moments per time" | A methods paper, novelty "moderate, subject to F-3" | Not a methods paper on its own. Either a section of a software paper on the engine, with scale results, or a short note for the federated community on the bandwidth saving. Both must cite coxdev, WebDISCO and H2O-3 and claim only D |
| Section 3.4, first row | "A new combination, subject to the search" | A combination; D is the only part not found |

## 5. Limits of this search

- Web search and four documents read in full or in part; most sources were read through search
  excerpts, not end to end. FedECA's appendix, Cyclops's source, the `survival` package's
  counting-process code and SAS's FAST option were not read.
- coxdev's repository may already contain blockwise or streaming use of its operator; only its
  manual, description and vignette excerpts were read.
- No patent search.
- Absence in a search is not proof of novelty. Before any claim in print, write to the coxdev
  authors or post the note as a preprint and ask.

## 6. References

- Andreux M, Manoel A, Menuet R, Saillard C, Simpson C (2020). Federated survival analysis with discrete-time Cox models. arXiv:2006.08997.
- Breheny P (2015). The Cox partial likelihood: score and Hessian. Lecture notes, BIOS 7210, University of Iowa. https://myweb.uiowa.edu/pbreheny/7210/f15/notes/10-27.pdf
- Carstensen B. Who needs the Cox model anyway? http://bendixcarstensen.com/WntCma.pdf
- coxstream. https://pypi.org/project/coxstream/
- Duan R, Luo C, Schuemie MJ, et al. (2020). Learning from local to global: an efficient distributed algorithm for modeling time-to-event data. JAMIA 27(7):1028-1036.
- glum documentation. The Cox proportional hazards model in glum. https://glum.readthedocs.io/en/latest/tutorials/cox_model/cox_model.html
- Her QL, et al. A distributed regression analysis application based on SAS software, part II: Cox proportional hazards regression. arXiv:1808.02392.
- Hu M, Shi X, Song PX-K (2024). Privacy enhanced collaborative inference in the Cox proportional hazards model for distributed data. arXiv:2409.04716.
- Lu C-L, Wang S, Ji Z, Wu Y, Xiong L, Jiang X, Ohno-Machado L (2015). WebDISCO: a web service for distributed Cox model learning without patient-level data sharing. JAMIA 22(6):1212-1219.
- Ogier du Terrail J, et al. (2023). FedECA: a federated external control arm method for causal inference with time-to-event data in distributed settings. arXiv:2311.16984.
- Taylor J, Hastie T, Narasimhan B. coxdev: Cox model deviance, gradient, information and score sampling. R package 0.4.0. https://github.com/jonathan-taylor/coxdev
- Taylor J, Hastie T, Narasimhan B, Tay K. Some notes on the score and Hessian of weighted Cox's proportional hazards with ties. glmnet vignette. https://glmnet.stanford.edu/articles/coxdev.html
- Whitehead J (1980). Fitting Cox's regression model to survival data using GLIM. Applied Statistics 29(3):268-275.

# Runs one fit with the R package pprof and writes the result as JSON. Called by pprof_r_adapter.py:
#   Rscript pprof_r_fit.R <dir> <family> <rows> <features> <strata> <ties> <tol> <max_iter>
# <dir> holds x.bin (doubles, column by column), provider.bin and y.bin or time.bin and event.bin.
# Base R and pprof only; statistics are pprof's.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]; family <- args[2]; n <- as.numeric(args[3]); p <- as.integer(args[4])
strata <- args[5]; ties <- args[6]; tol <- as.numeric(args[7]); max_iter <- as.integer(args[8])
suppressMessages(library(pprof))
clock <- function() proc.time()[["elapsed"]]
read <- function(name, count) readBin(file.path(dir, name), what = "double", n = count, size = 8, endian = "little")

t0 <- clock()
x <- matrix(read("x.bin", n * p), nrow = n, ncol = p)
covariates <- paste0("x", seq_len(p))
colnames(x) <- covariates
df <- as.data.frame(x)
rm(x)
df$provider <- as.integer(read("provider.bin", n))
if (family == "cox") {
  df$time <- read("time.bin", n)
  df$event <- read("event.bin", n)
} else {
  df$y <- read("y.bin", n)
}
t1 <- clock()
if (family == "cox") {
  d <- pp_data(df, x = covariates, provider = "provider", time = "time", event = "event",
               outcome = "survival", cutoff = 0, check = FALSE, message = FALSE)
  rm(df)
  t2 <- clock()
  fit <- fit_cox(d, strata = strata, ties = ties)
  stage <- fit$stage1
  if (!inherits(stage, "coxph") && !is.null(stage$fit)) stage <- stage$fit
  loglik <- stage$loglik[length(stage$loglik)]
  iterations <- fit$control$iterations[1]
  settings <- sprintf('{"strata":"%s","ties":"%s","cutoff":0,"stopping":"survival::coxph defaults"}', strata, ties)
} else {
  d <- pp_data(df, y = "y", x = covariates, provider = "provider", outcome = "binary",
               cutoff = 0, check = FALSE, message = FALSE)
  rm(df)
  t2 <- clock()
  fit <- fit_logistic_fe(d, algorithm = "SerBIN", tol = tol, max_iter = max_iter, bound = 10,
                         backtrack = TRUE, threads = 1, stop = "beta")
  loglik <- fit$loglik
  iterations <- fit$control$iterations
  settings <- sprintf('{"algorithm":"SerBIN","tol":%s,"max_iter":%d,"bound":10,"stop":"beta","threads":1,"cutoff":0}',
                      format(tol, digits = 17), max_iter)
}
t3 <- clock()
beta <- fit$coefficients$beta[covariates]
se <- sqrt(diag(vcov(fit)))[covariates]
memory <- gc()
number <- function(v) ifelse(is.finite(v), sprintf("%.17g", v), "null")
json <- sprintf(
  '{"coefficients":[%s],"standard_errors":[%s],"log_likelihood":%s,"iterations":%s,"converged":%s,"settings":%s,"seconds":{"read":%s,"prepare":%s,"fit":%s},"peak_r_mb":%s,"r_version":"%s","pprof_version":"%s","survival_version":"%s"}',
  paste(number(beta), collapse = ","), paste(number(se), collapse = ","), number(loglik),
  number(as.numeric(iterations)), if (isTRUE(fit$converged)) "true" else "false", settings,
  number(t1 - t0), number(t2 - t1), number(t3 - t2), number(sum(memory[, ncol(memory)])),
  paste(R.version$major, R.version$minor, sep = "."), as.character(packageVersion("pprof")),
  as.character(packageVersion("survival")))
writeLines(json, file.path(dir, "result.json"))

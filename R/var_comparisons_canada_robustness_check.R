################################################################################
# varp_levels_forecast_loop.R
# ---------------------------------------------------------------------------
# 1) Loops over p = 2..12 to fit a VAR(p) on differenced "Canada" data,
#    using five shrinkage methods:
#      - Ridge
#      - Nonparametric Shrinkage
#      - Normal prior (Stan)
#      - Lasso prior (Stan)
#      - Horseshoe prior (Stan)
#
# 2) For each p, produces 1-step-ahead forecasts on the differenced scale,
#    inverts to original (levels) scale, and computes RMSE & MAPE.
#
# 3) Collects all results in a data frame: columns (p, Method, RMSE, MAPE).
################################################################################


# -------------------------
# 1) Load required libraries
# -------------------------
library(vars)
library(VARshrink)
library(glmnet)
library(rstan)
library(tidyverse)

# Optional: speed up Stan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# -------------------------
# 2) Load the "Canada" data and define levels/differences
# -------------------------
data("Canada")
# We'll use the first 4 columns (e, prod, rw, U):
Y_levels <- as.matrix(Canada[, 1:4])  # the original (levels) data
colnames(Y_levels) <- c("e", "prod", "rw", "U")

# Differenced data for stationarity
Y_diff <- diff(Y_levels)

# Let Tfull_levels = total rows in levels
#     Tfull_diff   = total rows in differenced series
Tfull_levels <- nrow(Y_levels)
Tfull_diff   <- nrow(Y_diff)

# We hold out last 4 differences for testing => final 4 rows of Y_diff
Ttrain_diff <- Tfull_diff - 4

Ytrain_diff <- Y_diff[1:Ttrain_diff, , drop = FALSE]
Ytest_diff  <- Y_diff[(Ttrain_diff + 1):Tfull_diff, , drop = FALSE]

# For evaluation on the *levels* scale, remember that
#   Y_diff[i, ] = Y_levels[i+1, ] - Y_levels[i, ].
# So the test portion in levels is:
Ytrain_levels <- Y_levels[1:(Ttrain_diff + 1), , drop = FALSE]
Ytest_levels  <- Y_levels[(Ttrain_diff + 1):Tfull_levels, , drop = FALSE]

# The actual test *levels* for computing errors:
#   Ytest_levels[2:(Ttest_diff + 1), ]  (since the first row is the 'initial condition')
Ttest_diff       <- nrow(Ytest_diff)
actual_test_vals <- Ytest_levels[2:(Ttest_diff + 1), , drop = FALSE]

cat("Test set in levels has", nrow(Ytest_levels), "rows.\n")
cat("Test set in differences has", nrow(Ytest_diff), "rows.\n")


# -------------------------
# 3) Helper functions
# -------------------------

# (a) Build design matrix for a VAR(p) on *differenced* data
make_VAR_design_p <- function(Y, p) {
  # Y: T x d (already differenced)
  # p: number of lags
  # returns a list with
  #   X: (T - p) x (d*p)
  #   Y_out: (T - p) x d
  T <- nrow(Y)
  d <- ncol(Y)

  X <- matrix(NA_real_, nrow = T - p, ncol = d * p)
  Y_out <- Y[(p + 1):T, , drop = FALSE]

  for (t in (p + 1):T) {
    row_idx <- t - p
    lag_vec <- c()
    for (lag_i in 1:p) {
      lag_vec <- c(lag_vec, Y[t - lag_i, ])
    }
    X[row_idx, ] <- lag_vec
  }

  list(X = X, Y = Y_out)
}

# (b) Compute RMSE & MAPE
get_accuracy_metrics <- function(actual, predicted) {
  # 'actual' and 'predicted' both T x d
  diff_vals   <- as.numeric(actual - predicted)
  actual_vals <- as.numeric(actual)

  rmse <- sqrt(mean(diff_vals^2))

  # For MAPE, ignore near-zero actuals
  nonzero_idx <- which(abs(actual_vals) > 1e-8)
  mape <- NA_real_
  if (length(nonzero_idx) > 0) {
    mape <- 100 * mean(abs(diff_vals[nonzero_idx] / actual_vals[nonzero_idx]))
  }
  return(list(RMSE = rmse, MAPE = mape))
}

# (c) Forecast function: differenced domain -> levels
forecast_VARp_on_levels <- function(B, Ytrain_diff, Ytrain_levels, Ytest_diff, p) {
  # B: (d x d*p), fitted on differenced data
  # Ytrain_diff: (Ttrain_diff x d)
  # Ytest_diff: (Ttest_diff x d)
  # p: integer
  #
  # Returns: (Ttest_diff x d) matrix of predictions on the *levels* scale

  d <- ncol(Ytrain_diff)
  Ttest_diff <- nrow(Ytest_diff)

  # Combine the last p rows of training diffs with the test diffs
  Y_for_forecast <- rbind(tail(Ytrain_diff, p), Ytest_diff)

  preds_levels <- matrix(NA_real_, nrow = Ttest_diff, ncol = d)

  # Start from the last known "actual" level in training:
  current_level <- tail(Ytrain_levels, 1)  # shape: 1 x d

  # 1-step forecast in differenced space
  pred_one_step_diff <- function(B, y_lags) {
    # y_lags: (1 x (d*p))
    return(y_lags %*% t(B))  # => 1 x d
  }

  for (i in seq_len(Ttest_diff)) {
    row_idx <- p + i - 1
    lag_vec <- c()
    for (lag_j in 0:(p - 1)) {
      lag_vec <- c(lag_vec, Y_for_forecast[row_idx - lag_j, ])
    }
    lag_vec <- matrix(lag_vec, nrow = 1)

    pred_diff  <- pred_one_step_diff(B, lag_vec)   # 1 x d
    pred_level <- current_level + pred_diff        # 1 x d

    preds_levels[i, ] <- pred_level
    current_level     <- pred_level
  }
  preds_levels
}


# -------------------------
# 4) For each p in 2..12, fit the 5 methods, compute forecast errors
# -------------------------
all_results <- list()

for (pval in 3:15) {
  cat("\n-----------------------------\n")
  cat("Fitting VAR(p) with p =", pval, "\n")
  cat("-----------------------------\n")

  # Build design
  train_data <- make_VAR_design_p(Ytrain_diff, pval)
  X_train <- train_data$X  # (Ttrain_diff - pval) x (d*pval)
  Y_train <- train_data$Y  # (Ttrain_diff - pval) x d
  d <- ncol(Y_train)

  # Only proceed if there's enough training data
  if (nrow(X_train) <= 0) {
    cat("Skipping p =", pval, "because Ttrain_diff - pval <= 0\n")
    next
  }

  # (a) Fit Ridge
  lambda_ridge <- 0.1
  ridge_fit <- glmnet(
    x = X_train,
    y = Y_train,
    alpha = 0,  # ridge
    family = "mgaussian",
    lambda = lambda_ridge,
    intercept = FALSE
  )
  coef_ridge <- coef(ridge_fit, s = lambda_ridge)
  Bhat_ridge <- matrix(NA, nrow = d, ncol = d*pval)
  for (j in seq_len(d)) {
    Bhat_ridge[j, ] <- as.numeric(coef_ridge[[j]])[2:(d*pval + 1)]
  }

  # (b) NonparamShrink
  fit_ns <- VARshrink(
    y = Y_train,
    p = pval,
    type = "none",
    method = "ns"
  )
  varlist <- fit_ns$varresult
  Bhat_ns <- matrix(NA, nrow = d, ncol = d*pval)
  for (j in seq_len(d)) {
    Bhat_ns[j, ] <- varlist[[j]]$coefficients
  }

  # (c) Bayesian Normal
  stan_data_normal <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train,
    prior_scale = 1.0
  )
  fit_normal <- stan(
    file = "stan/var_normal.stan",
    data = stan_data_normal,
    iter = 2000,   # could increase if you have time
    warmup = 1000,
    chains = 4,
    seed = 123,
    control = list(adapt_delta = 0.9, max_treedepth = 12)
  )
  post_normal <- rstan::extract(fit_normal, pars = "B")
  Bhat_normal <- apply(post_normal$B, c(2,3), mean)

  # (d) Bayesian Lasso
  stan_data_lasso <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train,
    lambda = 1.0
  )
  fit_lasso <- stan(
    file = "stan/var_lasso.stan",
    data = stan_data_lasso,
    iter = 2000,
    warmup = 1000,
    chains = 4,
    seed = 123,
    control = list(adapt_delta = 0.9, max_treedepth = 12)
  )
  post_lasso <- rstan::extract(fit_lasso, pars = "B")
  Bhat_lasso <- apply(post_lasso$B, c(2,3), mean)

  # (e) Bayesian Horseshoe
  stan_data_hs <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train
  )
  fit_hs <- stan(
    file = "stan/var_horseshoe.stan",
    data = stan_data_hs,
    iter = 2000,
    warmup = 1000,
    chains = 4,
    seed = 123,
    control = list(adapt_delta = 0.9, max_treedepth = 12)
  )
  post_hs <- rstan::extract(fit_hs, pars = "B")
  Bhat_hs <- apply(post_hs$B, c(2,3), mean)

  # Evaluate
  method_list <- list(
    Ridge           = Bhat_ridge,
    NonparamShrink  = Bhat_ns,
    Normal          = Bhat_normal,
    Lasso           = Bhat_lasso,
    Horseshoe       = Bhat_hs
  )

  for (meth in names(method_list)) {
    Bmat <- method_list[[meth]]

    # 1) Forecast
    preds_levels <- forecast_VARp_on_levels(
      Bmat,
      Ytrain_diff = Ytrain_diff,
      Ytrain_levels = Ytrain_levels,
      Ytest_diff = Ytest_diff,
      p = pval
    )

    # 2) Accuracy
    metrics <- get_accuracy_metrics(actual_test_vals, preds_levels)

    # 3) Store
    one_row <- data.frame(
      p = pval,
      Method = meth,
      RMSE_onLevels = metrics$RMSE,
      MAPE_onLevels = metrics$MAPE
    )
    all_results[[length(all_results) + 1]] <- one_row
  }
}

# -------------------------
# 5) Compile results into one data frame
# -------------------------
final_results <- bind_rows(all_results)
cat("\n=== Final Results ===\n")
print(final_results)


final_results %>% group_by(Method) %>% summarise(mean_rmse=mean(RMSE_onLevels),sd_rmse=sd(RMSE_onLevels),mean_mape=mean(MAPE_onLevels),sd_mape=sd(MAPE_onLevels))

# Example usage:
#   - final_results %>% group_by(p, Method) %>% summarize(...)
#   - or plot how RMSE/ MAPE changes with p for each method.

# If you like, you can do:
#   final_results %>%
#     ggplot(aes(x = p, y = RMSE_onLevels, color = Method)) +
#     geom_line() +
#     geom_point()

cat("\nDone! Stored RMSE and MAPE for p = 2..12 in `final_results`.\n")

################################################################################
# varp_levels_forecast_loop_series_overall_1step.R
# -----------------------------------------------------------------------------
# 1) Loops over p = 1..10 to fit a VAR(p) on differenced "Canada" data,
#    using these methods:
#      - Ridge
#      - Nonparametric Shrinkage ("NS") via predict(fit_ns, ...)
#      - Normal prior (Stan)
#      - Lasso prior (Stan)
#      - Horseshoe prior (Stan)
#
# 2) For each p, produces 4 one-step-ahead forecasts on the levels scale
#    using a rolling (recursive) scheme:
#      - At forecast time i, the model uses the *actual* past differences
#        (from training or from previously observed test periods) to
#        predict the next period’s level.
#
# 3) Computes RMSE & MAPE per variable plus an overall “All” row.
#
# 4) Collects all results in a data frame:
#    (p, Method, variable, RMSE, MAPE)
#    where "variable" can be "e", "prod", "rw", "U", or "All".
#
# 5) Produces plots of RMSE and MAPE for each of the 4 variables in a 2x2 grid,
#    plus separate plots for the overall (“All”) metrics.
################################################################################

# -------------------------
# 1) Load required libraries
# -------------------------
library(vars)
library(VARshrink)
library(glmnet)
library(rstan)
library(tidyverse)
library(patchwork)  # for multi-plot arrangement

# Optional: speed up Stan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# -------------------------
# 2) Load and define levels/differences
# -------------------------
data("Canada")
Y_levels <- as.matrix(Canada[, 1:4])
colnames(Y_levels) <- c("e", "prod", "rw", "U")

# Differenced data
Y_diff <- diff(Y_levels)

Tfull_levels <- nrow(Y_levels)
Tfull_diff   <- nrow(Y_diff)

# Train on all but last 4 diffs
Ttrain_diff <- Tfull_diff - 4
Ytrain_diff <- Y_diff[1:Ttrain_diff, , drop = FALSE]
# The test set in *differences* is the last 4 rows
Ytest_diff  <- Y_diff[(Ttrain_diff + 1):Tfull_diff, , drop = FALSE]

# Corresponding levels split
Ytrain_levels <- Y_levels[1:(Ttrain_diff + 1), , drop = FALSE]
Ytest_levels  <- Y_levels[(Ttrain_diff + 1):Tfull_levels, , drop = FALSE]

cat("\n=== Basic Info ===\n")
cat("Train diffs:", nrow(Ytrain_diff), "rows\n")
cat("Test diffs: ", nrow(Ytest_diff),  "rows\n")
cat("Train levels:", nrow(Ytrain_levels), "rows\n")
cat("Test levels: ", nrow(Ytest_levels),  "rows\n")

# If we have 4 test diffs, that means Ytest_levels has 5 rows.
# The final 4 of those are the ones we'll forecast (one-step-ahead).
# The "actual" test levels to compare:
# if Ytest_levels is shape (5 x 4), the last 4 are the forecast targets
Ttest_levels <- nrow(Ytest_levels) - 1  # should be 4
actual_test_vals <- Ytest_levels[2:(Ttest_levels + 1), , drop = FALSE]

# -------------------------
# 3) Helper functions
# -------------------------

# (A) Create design matrix for a VAR(p) on differenced data
make_VAR_design_p <- function(Y, p) {
  # Y: T x d (already differenced)
  # returns:
  #   X: (T - p) x (d*p)
  #   Y_out: (T - p) x d
  T <- nrow(Y)
  d <- ncol(Y)
  if (T - p <= 0) return(list(X = NULL, Y = NULL))

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

# (B) Rolling one-step-ahead forecast function
one_step_ahead_forecast <- function(B, Ytrain_diff, Ytrain_levels, Ytest_levels, p) {
  #
  # B: (d x d*p) coefficient matrix from differenced VAR(p)
  # Ytrain_diff: training diffs (Ttrain_diff x d)
  # Ytrain_levels: training levels (Ttrain_diff+1 x d)
  # Ytest_levels: test levels ((4+1) x d) => 5 rows if we have 4 diffs
  # p: integer
  #
  # We produce 4 one-step-ahead forecasts in a rolling manner:
  #   - We keep track of the last p actual differences in a buffer "diff_history"
  #   - We also track the current actual level (start from the last row of training)
  #   - For each step i=1..4:
  #       1) Build a lag vector from diff_history (with most recent diff first)
  #       2) Predict the next difference = lag_vec %*% t(B)
  #       3) Forecast level = current_level + predicted_diff
  #       4) Then update diff_history with the *actual* difference (the difference
  #          between the newly observed test level and the old current_level),
  #          and update current_level with that actual test level
  #
  # Returns: (4 x d) predicted levels (the "one-step-ahead" forecast for each step)
  #
  d <- ncol(Ytrain_diff)
  Ttest <- nrow(Ytest_levels) - 1  # 4 in this scenario
  preds <- matrix(NA_real_, nrow = Ttest, ncol = d)

  # We'll buffer the last p differences from training
  diff_history <- tail(Ytrain_diff, p)  # shape: p x d
  # The current "actual" level is the last row of training levels
  current_level <- tail(Ytrain_levels, 1)  # 1 x d

  for (i in seq_len(Ttest)) {
    # 1) Build the 1 x (d*p) lag vector in reverse order
    #    i.e. diff_history[p, ] is the most recent difference
    #    We'll flatten it row-by-row
    lag_vec <- as.vector(t(diff_history[p:1, , drop = FALSE]))  # p * d elements
    lag_vec <- matrix(lag_vec, nrow=1)

    # 2) Predict next difference
    pred_diff <- lag_vec %*% t(B)  # 1 x d

    # 3) Forecast level = current_level + pred_diff
    pred_level <- current_level + pred_diff
    preds[i, ] <- pred_level

    # 4) Then we see the *actual* test level in row i+1 of Ytest_levels
    actual_next <- Ytest_levels[i+1, , drop=FALSE]  # shape: 1x d
    # The newly observed difference
    new_diff <- actual_next - current_level

    # Update diff_history (drop oldest row, append new_diff)
    if (p > 1) {
      diff_history <- rbind(diff_history[-1, , drop=FALSE], new_diff)
    } else {
      diff_history <- new_diff
    }
    # Update current_level to the *actual*, not the forecast
    current_level <- actual_next
  }
  colnames(preds) <- colnames(Ytrain_diff)
  preds
}

# (C) Compute per-series RMSE & MAPE
get_accuracy_metrics_by_series <- function(actual, predicted) {
  stopifnot(nrow(actual) == nrow(predicted), ncol(actual) == ncol(predicted))
  d <- ncol(actual)
  var_names <- colnames(actual)
  if (is.null(var_names)) var_names <- paste0("Var", seq_len(d))

  output_list <- vector("list", d)
  for(j in seq_len(d)) {
    a_j <- actual[, j]
    p_j <- predicted[, j]
    diff_j <- a_j - p_j
    rmse_j <- sqrt(mean(diff_j^2))
    nonzero_idx <- which(abs(a_j) > 1e-8)
    mape_j <- NA_real_
    if (length(nonzero_idx) > 0) {
      mape_j <- 100 * mean(abs(diff_j[nonzero_idx] / a_j[nonzero_idx]))
    }
    output_list[[j]] <- data.frame(
      variable = var_names[j],
      RMSE = rmse_j,
      MAPE = mape_j
    )
  }
  do.call(rbind, output_list)
}

# (D) Compute overall RMSE & MAPE
get_accuracy_metrics_overall <- function(actual, predicted) {
  diff_vals   <- as.numeric(actual - predicted)
  actual_vals <- as.numeric(actual)
  rmse <- sqrt(mean(diff_vals^2))
  nonzero_idx <- which(abs(actual_vals) > 1e-8)
  mape <- NA_real_
  if(length(nonzero_idx) > 0) {
    mape <- 100 * mean(abs(diff_vals[nonzero_idx]/actual_vals[nonzero_idx]))
  }
  data.frame(variable="All", RMSE=rmse, MAPE=mape)
}

# (E) Combine per-series + overall
get_accuracy_metrics_by_series_and_overall <- function(actual, predicted) {
  df_series <- get_accuracy_metrics_by_series(actual, predicted)
  df_all    <- get_accuracy_metrics_overall(actual, predicted)
  bind_rows(df_series, df_all)
}

# -------------------------
# 4) Main loop: p = 1..10
# -------------------------
all_results <- list()

for (pval in 1:12) {
  cat("\n=============================\n")
  cat("Fitting VAR(p) with p =", pval, "\n")
  cat("=============================\n")

  # 1) Build design matrix for training
  train_data <- make_VAR_design_p(Ytrain_diff, pval)
  X_train <- train_data$X
  Y_train <- train_data$Y
  if(is.null(X_train) || nrow(X_train) <= 0) {
    cat("Skipping p=", pval, " (not enough training rows)\n")
    next
  }
  d <- ncol(Y_train)

  ############################
  # (a) Ridge
  ############################
  lambda_ridge <- 0.1
  ridge_fit <- glmnet(
    x = X_train,
    y = Y_train,
    alpha = 0,  # ridge
    family="mgaussian",
    lambda=lambda_ridge,
    intercept=FALSE
  )
  coef_ridge <- coef(ridge_fit, s=lambda_ridge)
  Bhat_ridge <- matrix(NA_real_, nrow=d, ncol=d*pval)
  for(j in seq_len(d)) {
    Bhat_ridge[j, ] <- as.numeric(coef_ridge[[j]])[2:(d*pval + 1)]
  }

  preds_ridge <- one_step_ahead_forecast(
    Bhat_ridge, Ytrain_diff, Ytrain_levels, Ytest_levels, pval
  )
  df_ridge <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_ridge) %>%
    mutate(p=pval, Method="Ridge")

  ############################
  # (b) Nonparametric Shrinkage
  ############################
  fit_ns <- VARshrink(
    y    = Ytrain_diff,
    p    = pval,
    type = "none",
    method="ns"
  )
  # Extract the coefficient matrix from fit_ns
  # (the 1-step-ahead forecast is done manually in one_step_ahead_forecast)
  Bhat_ns <- matrix(NA_real_, nrow=d, ncol=d*pval)
  for (jj in seq_len(d)) {
    Bhat_ns[jj, ] <- fit_ns$varresult[[jj]]$coefficients
  }

  preds_ns <- one_step_ahead_forecast(
    Bhat_ns, Ytrain_diff, Ytrain_levels, Ytest_levels, pval
  )
  df_ns <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_ns) %>%
    mutate(p=pval, Method="NS")

  ############################
  # (c) Normal prior (Stan)
  ############################
  stan_data_normal <- list(
    T           = nrow(X_train),
    d           = d,
    p           = pval,
    X           = X_train,
    Y           = Y_train,
    prior_scale = 1.0
  )
  fit_normal <- stan(
    file   = "helicon/stan/var_normal.stan",
    data   = stan_data_normal,
    iter   = 2000,
    warmup = 1000,
    chains = 2,
    seed   = 123
  )
  post_normal <- rstan::extract(fit_normal, "B")
  Bhat_normal <- apply(post_normal$B, c(2,3), mean)

  preds_normal <- one_step_ahead_forecast(
    Bhat_normal, Ytrain_diff, Ytrain_levels, Ytest_levels, pval
  )
  df_normal <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_normal) %>%
    mutate(p=pval, Method="Normal")

  ############################
  # (d) Lasso prior (Stan)
  ############################
  stan_data_lasso <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train,
    lambda = 1.0
  )
  fit_lasso <- stan(
    file   = "helicon/stan/var_lasso.stan",
    data   = stan_data_lasso,
    iter   = 2000,
    warmup = 1000,
    chains = 2,
    seed   = 123
  )
  post_lasso <- rstan::extract(fit_lasso, "B")
  Bhat_lasso <- apply(post_lasso$B, c(2,3), mean)

  preds_lasso <- one_step_ahead_forecast(
    Bhat_lasso, Ytrain_diff, Ytrain_levels, Ytest_levels, pval
  )
  df_lasso <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_lasso) %>%
    mutate(p=pval, Method="Lasso")

  ############################
  # (e) Horseshoe prior (Stan)
  ############################
  stan_data_hs <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train
  )
  fit_hs <- stan(
    file   = "helicon/stan/var_horseshoe.stan",
    data   = stan_data_hs,
    iter   = 2000,
    warmup = 1000,
    chains = 2,
    seed   = 123
  )
  post_hs <- rstan::extract(fit_hs, "B")
  Bhat_hs <- apply(post_hs$B, c(2,3), mean)

  preds_hs <- one_step_ahead_forecast(
    Bhat_hs, Ytrain_diff, Ytrain_levels, Ytest_levels, pval
  )
  df_hs <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_hs) %>%
    mutate(p=pval, Method="Horseshoe")

  # Combine for this p
  df_pval <- bind_rows(df_ridge, df_ns, df_normal, df_lasso, df_hs)
  all_results[[length(all_results) + 1]] <- df_pval
}

# -------------------------
# 5) Combine into final_results
# -------------------------
final_results <- bind_rows(all_results)
cat("\n=== HEAD of final_results ===\n")
print(head(final_results, 20))

# Summaries
summary_stats <- final_results %>%
  group_by(Method, variable) %>%
  summarise(
    mean_rmse = mean(RMSE),
    sd_rmse   = sd(RMSE),
    mean_mape = mean(MAPE),
    sd_mape   = sd(MAPE),
    .groups   = "drop"
  )
cat("\n=== Summary Stats by Method, Variable ===\n")
print(summary_stats, n=40)

cat("\n=== Summary Stats by Method Only ===\n")
print(
  final_results %>%
    group_by(Method) %>%
    summarise(
      mean_rmse = mean(RMSE),
      sd_rmse   = sd(RMSE),
      mean_mape = mean(MAPE),
      sd_mape   = sd(MAPE),
      .groups = "drop"
    )
)

# -------------------------
# 6) Plots
# -------------------------
var_list <- c("e", "prod", "rw", "U", "All")

make_plot <- function(data, metric=c("RMSE","MAPE")) {
  metric <- match.arg(metric)
  ggplot(data, aes(x=p, y=.data[[metric]], color=Method)) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks=1:12, minor_breaks=NULL) +
    theme_minimal() +
    labs(x="p", y=metric)
}

rmse_plots <- list()
mape_plots <- list()

for (v in var_list) {
  df_v <- final_results %>% filter(variable == v)
  title_rmse <- if (v=="All") "RMSE on 1-step-ahead test set by method and p" else paste("RMSE for", v)
  title_mape <- if (v=="All") "MAPE on 1-step-ahead test set by method and p" else paste("MAPE for", v)

  rmse_plots[[v]] <- make_plot(df_v, "RMSE") + labs(title=title_rmse)
  mape_plots[[v]] <- make_plot(df_v, "MAPE") + labs(title=title_mape)
}

# 2x2 for the four main variables
grid_rmse_4 <- (rmse_plots[["e"]] / rmse_plots[["prod"]]) | (rmse_plots[["rw"]] / rmse_plots[["U"]])
grid_mape_4 <- (mape_plots[["e"]] / mape_plots[["prod"]]) | (mape_plots[["rw"]] / mape_plots[["U"]])

cat("\n=== RMSE for e, prod, rw, U (1-step-ahead, 2x2) ===\n")
print(grid_rmse_4)

cat("\n=== MAPE for e, prod, rw, U (1-step-ahead, 2x2) ===\n")
print(grid_mape_4)

grid_rmse_4 / grid_mape_4

cat("\n=== Overall (All) ===\n")
cat("\n--- RMSE(All) ---\n"); print(rmse_plots[["All"]])
cat("\n--- MAPE(All) ---\n"); print(mape_plots[["All"]])

cat("\nDone! Now using 4 rolling 1-step-ahead forecasts, updated with actual test data at each step.\n")
(rmse_plots[["All"]]) + (mape_plots[["All"]])

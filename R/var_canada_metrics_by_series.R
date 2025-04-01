################################################################################
# varp_levels_forecast_loop_series_overall.R
# -----------------------------------------------------------------------------
# 1) Loops over p = 1..10 to fit a VAR(p) on differenced "Canada" data,
#    using these methods:
#      - Ridge
#      - Nonparametric Shrinkage ("NS") via predict(fit_ns, ...)
#      - Normal prior (Stan)
#      - Lasso prior (Stan)
#      - Horseshoe prior (Stan)
#
# 2) For each p, produces 4-step-ahead forecasts on the differenced scale,
#    inverts to the original (levels) scale, and computes RMSE & MAPE
#    *per variable* + an "All" row (across all variables).
#
# 3) Collects all results in a data frame:
#    (p, Method, variable, RMSE, MAPE)
#    => "variable" can be "e", "prod", "rw", "U", or "All".
#
# 4) Produces plots of RMSE and MAPE for each of the 4 variables in a 2x2 grid,
#    plus separate plots for "All" with the requested titles.
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
Ytest_diff  <- Y_diff[(Ttrain_diff + 1):Tfull_diff, , drop = FALSE]

# Corresponding levels split
Ytrain_levels <- Y_levels[1:(Ttrain_diff + 1), , drop = FALSE]
Ytest_levels  <- Y_levels[(Ttrain_diff + 1):Tfull_levels, , drop = FALSE]

Ttest_diff       <- nrow(Ytest_diff)  # should be 4
actual_test_vals <- Ytest_levels[2:(Ttest_diff + 1), , drop = FALSE]

cat("\n=== Basic Info ===\n")
cat("Train diffs:", nrow(Ytrain_diff), "rows\n")
cat("Test diffs: ", nrow(Ytest_diff),  "rows\n")
cat("Train levels:", nrow(Ytrain_levels), "rows\n")
cat("Test levels: ", nrow(Ytest_levels),  "rows\n")

# -------------------------
# 3) Helper functions
# -------------------------
make_VAR_design_p <- function(Y, p) {
  # Y: T x d (differenced)
  # returns X: (T - p) x (d*p), Y_out: (T - p) x d
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

forecast_VARp_on_levels <- function(B, Ytrain_diff, Ytrain_levels, Ytest_diff, p) {
  # B: (d x (d*p)), fitted on differenced data
  # return: (Ttest_diff x d) forecasts on levels
  d <- ncol(Ytrain_diff)
  Ttest_diff <- nrow(Ytest_diff)

  Y_for_forecast <- rbind(tail(Ytrain_diff, p), Ytest_diff)
  preds_levels <- matrix(NA_real_, nrow = Ttest_diff, ncol = d)
  current_level <- tail(Ytrain_levels, 1)

  pred_one_step_diff <- function(B, y_lags) y_lags %*% t(B)

  for (i in seq_len(Ttest_diff)) {
    row_idx <- p + i - 1
    lag_vec <- c()
    for (lag_j in 0:(p - 1)) {
      lag_vec <- c(lag_vec, Y_for_forecast[row_idx - lag_j, ])
    }
    lag_vec <- matrix(lag_vec, nrow = 1)

    pred_diff  <- pred_one_step_diff(B, lag_vec)
    pred_level <- current_level + pred_diff
    preds_levels[i, ] <- pred_level
    current_level     <- pred_level
  }
  colnames(preds_levels) <- colnames(Ytrain_diff)
  preds_levels
}

# (A) Per-series RMSE & MAPE
get_accuracy_metrics_by_series <- function(actual, predicted) {
  stopifnot(nrow(actual) == nrow(predicted),
            ncol(actual) == ncol(predicted))

  d <- ncol(actual)
  var_names <- colnames(actual)
  if (is.null(var_names)) {
    var_names <- paste0("Var", seq_len(d))
  }

  results_list <- vector("list", d)
  for (j in seq_len(d)) {
    a_j <- actual[, j]
    p_j <- predicted[, j]
    diff_j <- a_j - p_j

    rmse_j <- sqrt(mean(diff_j^2, na.rm=TRUE))

    # MAPE
    nonzero_idx <- which(abs(a_j) > 1e-8)
    mape_j <- NA_real_
    if (length(nonzero_idx) > 0) {
      mape_j <- 100 * mean(abs(diff_j[nonzero_idx] / a_j[nonzero_idx]))
    }

    results_list[[j]] <- data.frame(
      variable = var_names[j],
      RMSE = rmse_j,
      MAPE = mape_j
    )
  }
  do.call(rbind, results_list)
}

# (B) Overall RMSE & MAPE (across all series)
get_accuracy_metrics_overall <- function(actual, predicted) {
  # Flatten across all T*d entries
  diff_vals   <- as.numeric(actual - predicted)
  actual_vals <- as.numeric(actual)

  rmse <- sqrt(mean(diff_vals^2, na.rm=TRUE))

  nonzero_idx <- which(abs(actual_vals) > 1e-8)
  mape <- NA_real_
  if (length(nonzero_idx) > 0) {
    mape <- 100 * mean(abs(diff_vals[nonzero_idx] / actual_vals[nonzero_idx]))
  }

  data.frame(variable = "All", RMSE = rmse, MAPE = mape)
}

# Combined function that returns *both* per-series and overall
get_accuracy_metrics_by_series_and_overall <- function(actual, predicted) {
  df_series  <- get_accuracy_metrics_by_series(actual, predicted)
  df_all     <- get_accuracy_metrics_overall(actual, predicted)
  dplyr::bind_rows(df_series, df_all)
}

# -------------------------
# 4) Main loop: p = 1..10
# -------------------------
all_results <- list()

for (pval in 1:10) {
  cat("\n=============================\n")
  cat("Fitting VAR(p) with p =", pval, "\n")
  cat("=============================\n")

  # Build design
  train_data <- make_VAR_design_p(Ytrain_diff, pval)
  X_train <- train_data$X
  Y_train <- train_data$Y

  if (is.null(X_train) || nrow(X_train) <= 0) {
    cat("Skipping p =", pval, "because not enough training data\n")
    next
  }
  d <- ncol(Y_train)

  ######################################################
  # (a) Ridge
  ######################################################
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

  preds_levels_ridge <- forecast_VARp_on_levels(
    B = Bhat_ridge,
    Ytrain_diff   = Ytrain_diff,
    Ytrain_levels = Ytrain_levels,
    Ytest_diff    = Ytest_diff,
    p = pval
  )

  df_ridge <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_levels_ridge) %>%
    mutate(p = pval, Method = "Ridge") %>%
    select(p, Method, variable, RMSE, MAPE)

  ######################################################
  # (b) NS (predict)
  ######################################################
  fit_ns <- VARshrink(
    y = Ytrain_diff,
    p = pval,
    type = "none",
    method = "ns"
  )
  pred_out_ns <- predict(fit_ns, n.ahead = Ttest_diff)
  var_names <- colnames(Ytrain_diff)

  pred_diff_mat_ns <- matrix(NA_real_, nrow = Ttest_diff, ncol = d,
                             dimnames = list(NULL, var_names))
  for (j in seq_len(d)) {
    var_j <- var_names[j]
    pred_diff_mat_ns[, j] <- pred_out_ns$fcst[[var_j]][, "fcst"]
  }

  preds_levels_ns <- matrix(NA_real_, nrow = Ttest_diff, ncol = d,
                            dimnames = list(NULL, var_names))
  current_level <- tail(Ytrain_levels, 1)
  for (i in seq_len(Ttest_diff)) {
    preds_levels_ns[i, ] <- current_level + pred_diff_mat_ns[i, ]
    current_level        <- preds_levels_ns[i, ]
  }

  df_ns <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_levels_ns) %>%
    mutate(p = pval, Method = "NS") %>%
    select(p, Method, variable, RMSE, MAPE)

  ######################################################
  # (c) Normal prior (Stan)
  ######################################################
  stan_data_normal <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train,
    prior_scale = 1.0
  )
  fit_normal <- stan(
    file = "helicon/stan/var_normal.stan",
    data = stan_data_normal,
    iter = 2000,
    warmup = 1000,
    chains = 4,
    seed = 123,
    control = list(adapt_delta = 0.9, max_treedepth = 12)
  )
  post_normal <- rstan::extract(fit_normal, pars = "B")
  Bhat_normal <- apply(post_normal$B, c(2,3), mean)

  preds_levels_normal <- forecast_VARp_on_levels(
    B = Bhat_normal,
    Ytrain_diff   = Ytrain_diff,
    Ytrain_levels = Ytrain_levels,
    Ytest_diff    = Ytest_diff,
    p = pval
  )

  df_normal <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_levels_normal) %>%
    mutate(p = pval, Method = "Normal") %>%
    select(p, Method, variable, RMSE, MAPE)

  ######################################################
  # (d) Lasso
  ######################################################
  stan_data_lasso <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train,
    lambda = 1.0
  )
  fit_lasso <- stan(
    file = "helicon/stan/var_lasso.stan",
    data = stan_data_lasso,
    iter = 2000,
    warmup = 1000,
    chains = 4,
    seed = 123,
    control = list(adapt_delta = 0.9, max_treedepth = 12)
  )
  post_lasso <- rstan::extract(fit_lasso, pars = "B")
  Bhat_lasso <- apply(post_lasso$B, c(2,3), mean)

  preds_levels_lasso <- forecast_VARp_on_levels(
    B = Bhat_lasso,
    Ytrain_diff   = Ytrain_diff,
    Ytrain_levels = Ytrain_levels,
    Ytest_diff    = Ytest_diff,
    p = pval
  )

  df_lasso <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_levels_lasso) %>%
    mutate(p = pval, Method = "Lasso") %>%
    select(p, Method, variable, RMSE, MAPE)

  ######################################################
  # (e) Horseshoe
  ######################################################
  stan_data_hs <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train
  )
  fit_hs <- stan(
    file = "helicon/stan/var_horseshoe.stan",
    data = stan_data_hs,
    iter = 2000,
    warmup = 1000,
    chains = 4,
    seed = 123,
    control = list(adapt_delta = 0.9, max_treedepth = 12)
  )
  post_hs <- rstan::extract(fit_hs, pars = "B")
  Bhat_hs <- apply(post_hs$B, c(2,3), mean)

  preds_levels_hs <- forecast_VARp_on_levels(
    B = Bhat_hs,
    Ytrain_diff   = Ytrain_diff,
    Ytrain_levels = Ytrain_levels,
    Ytest_diff    = Ytest_diff,
    p = pval
  )

  df_hs <- get_accuracy_metrics_by_series_and_overall(actual_test_vals, preds_levels_hs) %>%
    mutate(p = pval, Method = "Horseshoe") %>%
    select(p, Method, variable, RMSE, MAPE)

  # Combine for this p
  df_pval <- bind_rows(df_ridge, df_ns, df_normal, df_lasso, df_hs)
  all_results[[length(all_results)+1]] <- df_pval
}

# -------------------------
# 5) Combine into final_results
# -------------------------
final_results <- bind_rows(all_results)
cat("\n=== HEAD of final_results ===\n")
print(head(final_results, 20))

# Example summary for "All" vs. each variable
summary_stats <- final_results %>%
  group_by(Method, variable) %>%
  summarise(
    mean_rmse = mean(RMSE),
    sd_rmse   = sd(RMSE),
    mean_mape = mean(MAPE),
    sd_mape   = sd(MAPE),
    .groups = "drop"
  )
cat("\n=== Summary Stats ===\n")
print(summary_stats,n=40)





final_results %>%
  group_by(Method) %>%
  summarise(
    mean_rmse = mean(RMSE),
    sd_rmse   = sd(RMSE),
    mean_mape = mean(MAPE),
    sd_mape   = sd(MAPE),
    .groups = "drop"
  )


cat("\n=== Summary Stats ===\n")
print(summary_stats,n=40)

# -------------------------
# 6) Plots
# -------------------------
library(patchwork)

# We'll keep var_list in the order: e, prod, rw, U, All
var_list <- c("e", "prod", "rw", "U", "All")

# We'll do 2x2 for the four main variables, and then a separate plot for "All"

# (A) create a plot function that we can reuse
make_plot <- function(data, metric = c("RMSE","MAPE")) {
  metric <- match.arg(metric)
  ggplot(data, aes(x = p, y = .data[[metric]], color = Method)) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks = seq(1, 10, by = 1), minor_breaks = NULL) +
    theme_minimal() +
    labs(x = "p", y = metric)
}

rmse_plots <- list()
mape_plots <- list()

for (v in var_list) {
  # filter data for variable v
  df_v <- final_results %>% filter(variable == v)

  # Default title
  title_rmse <- paste("RMSE for", v)
  title_mape <- paste("MAPE for", v)

  # If variable == "All", override titles:
  if (v == "All") {
    title_rmse <- "RMSE on test set by method and p"
    title_mape <- "MAPE on test set by method and p"
  }

  p_rmse <- make_plot(df_v, "RMSE") + labs(title = title_rmse)
  p_mape <- make_plot(df_v, "MAPE") + labs(title = title_mape)

  rmse_plots[[v]] <- p_rmse
  mape_plots[[v]] <- p_mape
}

# (B) Combine the 4 variable plots in 2x2 and show the "All" variable separately
cat("\n=== Plots for e, prod, rw, U (2x2) ===\n")

grid_rmse_4 <- (rmse_plots[["e"]] / rmse_plots[["prod"]]) | (rmse_plots[["rw"]] / rmse_plots[["U"]])
grid_mape_4 <- (mape_plots[["e"]] / mape_plots[["prod"]]) | (mape_plots[["rw"]] / mape_plots[["U"]])

cat("\n--- RMSE for e,prod,rw,U ---\n")
print(grid_rmse_4)

cat("\n--- MAPE for e,prod,rw,U ---\n")
print(grid_mape_4)

cat("\n=== Plot for All variables combined ===\n")
cat("--- RMSE(All) ---\n")
print(rmse_plots[["All"]])

cat("\n--- MAPE(All) ---\n")
print(mape_plots[["All"]])

cat("\nDone!\n")


 (rmse_plots[["All"]])+
  (mape_plots[["All"]])

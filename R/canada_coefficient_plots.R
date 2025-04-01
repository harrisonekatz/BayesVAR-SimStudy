################################################################################
# varp_all_methods_forecast_ROLLING_1STEP.R
# -----------------------------------------------------------------------------
# Demonstrates 5 methods on the Canada data:
#   1) Ridge (glmnet)
#   2) Nonparametric Shrinkage (VARshrink)
#   3) Normal prior (Stan)
#   4) Lasso prior (Stan)
#   5) Horseshoe prior (Stan)
# for four different lags: 2, 5, 7, 10.
#
# BUT IMPORTANTLY: uses the same "rolling 1-step-ahead" forecast approach
# that Script (2) used:
#   - Each 1-step forecast is formed using the last p *actual* differences.
#   - Once the real next period is observed, we incorporate that difference
#     in the history before forecasting the next step.
#
# Steps:
#   1) Load "Canada" data (vars package), use columns (e, prod, rw, U).
#   2) Train on differenced data (all but last 4 obs).
#   3) For each p in {2,5,7,10}, fit & forecast each method, invert to levels
#      in a *rolling* manner, compute RMSE & MAPE, and store coefficient matrix.
#   4) Produce:
#       - Accuracy table
#       - 2x2 boxplot of coefficient distributions, faceted by lag p,
#         colored by method
################################################################################

# -------------------------
# 1) Load libraries
# -------------------------
library(vars)       # for data("Canada")
library(VARshrink)  # for Nonparametric Shrinkage
library(glmnet)     # for Ridge
library(rstan)      # for Stan models (Normal/Lasso/Horseshoe)
library(tidyverse)
library(ggplot2)

# Optional: speed up Stan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# -------------------------
# 2) Load and prepare data
# -------------------------
cat("Loading Canada data...\n")
data("Canada", package="vars")

# Use first 4 columns (e, prod, rw, U):
Y_levels <- as.matrix(Canada[, 1:4])
colnames(Y_levels) <- c("e", "prod", "rw", "U")

# Differenced data
Y_diff <- diff(Y_levels)

Tfull_levels <- nrow(Y_levels)
Tfull_diff   <- nrow(Y_diff)

# We hold out the last 4 differences for testing
Ttest_diff  <- 4
Ttrain_diff <- Tfull_diff - Ttest_diff

Ytrain_diff <- Y_diff[1:Ttrain_diff, , drop=FALSE]
Ytest_diff  <- Y_diff[(Ttrain_diff+1):Tfull_diff, , drop=FALSE]

# The corresponding levels
Ytrain_levels <- Y_levels[1:(Ttrain_diff+1), , drop=FALSE]
Ytest_levels  <- Y_levels[(Ttrain_diff+1):Tfull_levels, , drop=FALSE]

cat("Train diffs:", nrow(Ytrain_diff),
    "| Test diffs:", nrow(Ytest_diff), "\n")
cat("Train levels:", nrow(Ytrain_levels),
    "| Test levels:", nrow(Ytest_levels), "\n\n")

# The actual test "levels" we compare to are rows 2..5 of Ytest_levels
# (the first row is just the 'initial condition' for the test period)
actual_test_levels <- Ytest_levels[2:(Ttest_diff+1), , drop=FALSE]


# -------------------------
# 3) Helper functions
# -------------------------

# (A) Create design matrix for a VAR(p) on differenced data
make_VAR_design_p <- function(Y, p) {
  # Y: T x d
  T <- nrow(Y)
  d <- ncol(Y)
  
  if(T <= p) {
    return(list(X=NULL, Y=NULL))  # not enough data
  }
  
  X <- matrix(NA_real_, nrow=(T-p), ncol=d*p)
  Y_out <- Y[(p+1):T, , drop=FALSE]
  
  for(t in (p+1):T) {
    row_idx <- t - p
    lag_vec <- c()
    for(lag_i in 1:p) {
      lag_vec <- c(lag_vec, Y[t - lag_i, ])
    }
    X[row_idx, ] <- lag_vec
  }
  list(X=X, Y=Y_out)
}


# (B) Rolling 1-step-ahead forecast in levels
#     (Matches Script 2's approach exactly)
rolling_one_step_ahead_forecast <- function(B, Ytrain_diff, Ytrain_levels,
                                            Ytest_levels, p) {
  #
  # B:  (d x (d*p)) coefficient matrix (trained on differenced data)
  # Ytrain_diff:     Ttrain_diff x d
  # Ytrain_levels:   (Ttrain_diff+1) x d
  # Ytest_levels:    (4 + 1) x d   (the first row is the “initial condition”)
  # p: number of lags
  #
  # We produce 4 one-step-ahead forecasts, each time using the last p *actual*
  # differences and the last *actual* level.
  #
  d <- ncol(Ytrain_diff)
  Ttest <- nrow(Ytest_levels) - 1  # should be 4 here
  if(Ttest < 1) return(NULL)       # safety
  
  preds <- matrix(NA_real_, nrow = Ttest, ncol = d)
  colnames(preds) <- colnames(Ytrain_diff)
  
  # Buffer of last p *actual* diffs
  # (the most recent diff is row p of this buffer)
  diff_history <- tail(Ytrain_diff, p)   # p x d
  
  # Current "actual" level is the last row of training levels
  current_level <- tail(Ytrain_levels, 1) # shape 1 x d
  
  for (i in seq_len(Ttest)) {
    # 1) Build the lag vector from last p actual diffs
    #    with the most recent difference last in the order
    #    (common in a row-by-row flattening)
    #    We do e.g. diff_history[p, ] is the most recent
    #    so we can flatten from p down to 1:
    lag_vec <- as.vector(t(diff_history[p:1, , drop=FALSE]))
    lag_vec <- matrix(lag_vec, nrow=1)  # shape 1 x (d*p)
    
    # 2) Predict next difference
    pred_diff <- lag_vec %*% t(B) # (1 x d)
    
    # 3) The forecasted level = current *actual* level + predicted diff
    pred_level <- current_level + pred_diff
    preds[i, ] <- pred_level
    
    # 4) Now incorporate the *actual* test data for step i
    #    i.e. the row i+1 of Ytest_levels is the "true" new level
    actual_next_level <- Ytest_levels[i + 1, , drop=FALSE]
    # The actual difference:
    actual_diff_i <- actual_next_level - current_level
    
    # update diff_history: drop oldest row, add the new diff
    if(p > 1) {
      diff_history <- rbind(diff_history[-1, , drop=FALSE], actual_diff_i)
    } else {
      diff_history <- actual_diff_i
    }
    
    # update current_level to the *actual* level
    current_level <- actual_next_level
  }
  
  preds
}


# (C) Accuracy metrics
get_accuracy_metrics <- function(actual, predicted) {
  diff_vals   <- as.numeric(actual - predicted)
  actual_vals <- as.numeric(actual)
  rmse        <- sqrt(mean(diff_vals^2))
  
  nonzero_idx <- which(abs(actual_vals) > 1e-8)
  if(length(nonzero_idx) > 0) {
    mape <- 100 * mean(abs(diff_vals[nonzero_idx] / actual_vals[nonzero_idx]))
  } else {
    mape <- NA_real_
  }
  list(RMSE=rmse, MAPE=mape)
}


# -------------------------
# 4) Main loop over p in {2,5,7,10}
# -------------------------
p_values <- c(3,6,9,12)

all_accuracy <- list()
all_coefs    <- list()

for(pval in p_values) {
  cat("\n============================\n")
  cat("Fitting all methods for p =", pval, "\n")
  cat("============================\n")
  
  # Build design for the methods that need (X, Y)
  design_p <- make_VAR_design_p(Ytrain_diff, pval)
  X_train  <- design_p$X
  Y_train  <- design_p$Y
  
  if(is.null(X_train) || nrow(X_train) == 0) {
    cat("Skipping p=", pval, "due to insufficient data.\n")
    next
  }
  
  d <- ncol(Ytrain_diff)
  
  ##############################
  # 1) Ridge
  ##############################
  cat("  [Method: Ridge]\n")
  lambda_ridge <- 0.1
  ridge_fit <- glmnet(
    x         = X_train,
    y         = Y_train,
    alpha     = 0,           # alpha=0 => ridge
    family    = "mgaussian",
    lambda    = lambda_ridge,
    intercept = FALSE
  )
  
  # Extract B matrix: d x (d*pval)
  coef_ridge <- coef(ridge_fit, s=lambda_ridge)
  Bhat_ridge <- matrix(NA_real_, nrow=d, ncol=d*pval)
  for(j in seq_len(d)) {
    Bhat_ridge[j, ] <- as.numeric(coef_ridge[[j]])[2:(d*pval + 1)]
  }
  
  # Rolling 1-step-ahead forecast
  preds_ridge <- rolling_one_step_ahead_forecast(
    Bhat_ridge,
    Ytrain_diff,
    Ytrain_levels,
    Ytest_levels,
    pval
  )
  acc_ridge   <- get_accuracy_metrics(actual_test_levels, preds_ridge)
  
  coefs_ridge <- data.frame(
    Method   = "Ridge",
    Lag      = factor(pval),
    Value    = as.numeric(Bhat_ridge)
  )
  
  ##############################
  # 2) Nonparametric Shrinkage (NS)
  ##############################
  cat("  [Method: NS]\n")
  fit_ns <- VARshrink(
    y    = Ytrain_diff,
    p    = pval,
    type = "none",
    method = "ns"
  )
  # Extract B matrix from the fit
  Bhat_ns <- matrix(NA_real_, nrow=d, ncol=d*pval)
  for(jj in seq_len(d)) {
    Bhat_ns[jj, ] <- fit_ns$varresult[[jj]]$coefficients
  }
  
  # Rolling forecast using Bhat_ns
  preds_ns <- rolling_one_step_ahead_forecast(
    Bhat_ns,
    Ytrain_diff,
    Ytrain_levels,
    Ytest_levels,
    pval
  )
  acc_ns <- get_accuracy_metrics(actual_test_levels, preds_ns)
  
  coefs_ns <- data.frame(
    Method   = "NS",
    Lag      = factor(pval),
    Value    = as.numeric(Bhat_ns)
  )
  
  ##############################
  # 3) Normal prior (Stan)
  ##############################
  cat("  [Method: Normal]\n")
  stan_data_normal <- list(
    T           = nrow(X_train),
    d           = d,
    p           = pval,
    X           = X_train,
    Y           = Y_train,
    prior_scale = 1.0
  )
  fit_normal <- stan(
    file   = "helicon/stan/var_normal.stan",   # Adjust path if needed
    data   = stan_data_normal,
    iter   = 1000,
    warmup = 500,
    chains = 2,
    seed   = 123
  )
  post_normal <- rstan::extract(fit_normal, "B")
  Bhat_normal <- apply(post_normal$B, c(2,3), mean)
  
  preds_normal <- rolling_one_step_ahead_forecast(
    Bhat_normal,
    Ytrain_diff,
    Ytrain_levels,
    Ytest_levels,
    pval
  )
  acc_normal <- get_accuracy_metrics(actual_test_levels, preds_normal)
  
  coefs_normal <- data.frame(
    Method   = "Normal",
    Lag      = factor(pval),
    Value    = as.numeric(Bhat_normal)
  )
  
  ##############################
  # 4) Lasso prior (Stan)
  ##############################
  cat("  [Method: Lasso]\n")
  stan_data_lasso <- list(
    T      = nrow(X_train),
    d      = d,
    p      = pval,
    X      = X_train,
    Y      = Y_train,
    lambda = 1.0
  )
  fit_lasso <- stan(
    file   = "helicon/stan/var_lasso.stan",   # Adjust path if needed
    data   = stan_data_lasso,
    iter   = 1000,
    warmup = 500,
    chains = 2,
    seed   = 123
  )
  post_lasso <- rstan::extract(fit_lasso, "B")
  Bhat_lasso <- apply(post_lasso$B, c(2,3), mean)
  
  preds_lasso <- rolling_one_step_ahead_forecast(
    Bhat_lasso,
    Ytrain_diff,
    Ytrain_levels,
    Ytest_levels,
    pval
  )
  acc_lasso <- get_accuracy_metrics(actual_test_levels, preds_lasso)
  
  coefs_lasso <- data.frame(
    Method   = "Lasso",
    Lag      = factor(pval),
    Value    = as.numeric(Bhat_lasso)
  )
  
  ##############################
  # 5) Horseshoe prior (Stan)
  ##############################
  cat("  [Method: Horseshoe]\n")
  stan_data_hs <- list(
    T = nrow(X_train),
    d = d,
    p = pval,
    X = X_train,
    Y = Y_train
  )
  fit_hs <- stan(
    file   = "helicon/stan/var_horseshoe.stan",   # Adjust path if needed
    data   = stan_data_hs,
    iter   = 1000,
    warmup = 500,
    chains = 2,
    seed   = 123
  )
  post_hs <- rstan::extract(fit_hs, "B")
  Bhat_hs <- apply(post_hs$B, c(2,3), mean)
  
  preds_hs <- rolling_one_step_ahead_forecast(
    Bhat_hs,
    Ytrain_diff,
    Ytrain_levels,
    Ytest_levels,
    pval
  )
  acc_hs <- get_accuracy_metrics(actual_test_levels, preds_hs)
  
  coefs_hs <- data.frame(
    Method   = "Horseshoe",
    Lag      = factor(pval),
    Value    = as.numeric(Bhat_hs)
  )
  
  # Collect results for this p
  df_accuracy_p <- data.frame(
    Lag    = pval,
    Method = c("Ridge","NS","Normal","Lasso","Horseshoe"),
    RMSE   = c(acc_ridge$RMSE, acc_ns$RMSE, acc_normal$RMSE,
               acc_lasso$RMSE, acc_hs$RMSE),
    MAPE   = c(acc_ridge$MAPE, acc_ns$MAPE, acc_normal$MAPE,
               acc_lasso$MAPE, acc_hs$MAPE)
  )
  
  df_coefs_p <- bind_rows(
    coefs_ridge, coefs_ns, coefs_normal,
    coefs_lasso, coefs_hs
  )
  
  all_accuracy[[length(all_accuracy) + 1]] <- df_accuracy_p
  all_coefs[[length(all_coefs) + 1]]      <- df_coefs_p
}

# -------------------------
# 5) Combine & Print Results
# -------------------------
accuracy_table <- bind_rows(all_accuracy)
cat("\n=== Final Accuracy Table (Rolling 1-step) ===\n")
print(accuracy_table)

# Combine all coefficients
coef_table <- bind_rows(all_coefs)

# -------------------------
# 6) Plot coefficient distributions by method & p
# -------------------------
ggplot(coef_table, aes(x=Method, y=Value, color=Method)) +
  geom_boxplot(outlier.alpha=0.5) +
  facet_wrap(
    ~Lag,
    nrow=2, ncol=2,
    scales="free_x",
    labeller = labeller(Lag = function(x) paste0("p=", x))
  ) +
  labs(
    title="Coefficient Distributions by Method",
    x="Method",
    y="Coefficient Value"
  ) +
  theme_bw(base_size=12) +
  theme(legend.position="bottom")

cat("\nDone! All 5 methods now use the same rolling 1-step-ahead procedure, matching Script (2).\n")

################################################################################
# compare_VARp_methods_1step_rolling.R
# ------------------------------------------------------------------------------
# Illustrates how to fit a VAR(p) to differenced "Canada" data, using:
#   1) Ridge (frequentist)
#   2) Nonparametric Shrinkage (frequentist)
#   3) Normal prior (Bayesian) via Stan
#   4) Lasso prior (Bayesian) via Stan
#   5) Horseshoe prior (Bayesian) via Stan
#
# Then we do rolling 1-step-ahead forecasts on the *levels*, updating the
# "actual" differences after each step (same approach as your "script two").
################################################################################

# -------------------------
# 1) Load required libraries
# -------------------------
library(vars)
library(VARshrink)
library(glmnet)
library(rstan)
library(tidyverse)

# Speed up Stan (optional)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# -------------------------
# 2) Load and difference the data
# -------------------------
data("Canada")
Y_levels <- as.matrix(Canada[, 1:4])  # columns: e, prod, rw, U
colnames(Y_levels) <- c("e","prod","rw","U")

# We'll difference once (like script two).
Y_diff <- diff(Y_levels)

# Dimensions
Tfull_levels <- nrow(Y_levels)  # number of time points in levels
Tfull_diff   <- nrow(Y_diff)    # number of time points in differences
d <- ncol(Y_diff)               # dimension (4 for e, prod, rw, U)

# We'll hold out the last few rows for testing. For illustration, say last 4 diffs:
Ttest_diff <- 4
Ttrain_diff <- Tfull_diff - Ttest_diff
Ytrain_diff <- Y_diff[1:Ttrain_diff, , drop=FALSE]
Ytest_diff  <- Y_diff[(Ttrain_diff+1):Tfull_diff, , drop=FALSE]

# The corresponding levels we have:
#  - The "training levels" end exactly one row after Ytrain_diff
#  - The "test levels" start there
Ytrain_levels <- Y_levels[1:(Ttrain_diff + 1), , drop=FALSE]
Ytest_levels  <- Y_levels[(Ttrain_diff + 1):Tfull_levels, , drop=FALSE]

cat("Train diffs has", nrow(Ytrain_diff), "rows.\n")
cat("Test diffs has ", nrow(Ytest_diff), "rows.\n")
cat("Train levels has", nrow(Ytrain_levels), "rows.\n")
cat("Test levels has ", nrow(Ytest_levels), "rows.\n")

# The actual test portion on the *levels* side is 1-step-ahead forecasts for the
# final 4 time points, so Ytest_levels is (4+1)=5 rows in that scenario.
# The first row of Ytest_levels is effectively "initialization" for the test set,
# and the last 4 rows are the actual times we will forecast.

# -------------------------
# 3) Build design matrix for Ytrain_diff
#    (like your original 'make_VAR_design_p' function)
# -------------------------
make_VAR_design_p <- function(Y, p) {
  # Y: T x d matrix (already differenced)
  # p: number of lags
  # Returns:
  #   X: (T - p) x (d * p)
  #   Y_out: (T - p) x d
  T <- nrow(Y)
  d <- ncol(Y)
  if(T - p <= 0) return(list(X=NULL, Y=NULL))

  X <- matrix(NA_real_, nrow=T-p, ncol=d*p)
  Y_out <- Y[(p+1):T, , drop=FALSE]

  for(t in (p+1):T) {
    row_idx <- t - p
    # gather y_{t-1}, ..., y_{t-p} (each is dimension d) into one row
    lags <- c()
    for(lag_i in 1:p) {
      lags <- c(lags, Y[t - lag_i, ])
    }
    X[row_idx, ] <- lags
  }
  list(X=X, Y=Y_out)
}

# -------------------------
# 4) Rolling 1-step-ahead forecast function on the *levels*
# -------------------------
one_step_ahead_forecast <- function(B, Ytrain_diff, Ytrain_levels, Ytest_levels, p) {
  #
  # B: (d x (d*p)) coefficient matrix from the differenced VAR(p) model
  # Ytrain_diff: training diffs (Ttrain_diff x d)
  # Ytrain_levels: training levels (Ttrain_diff+1 x d)
  # Ytest_levels: test levels (4+1 x d) if 4 test diffs
  # p: integer lag
  #
  # Returns a (Ttest x d) matrix of predicted *levels*, where Ttest = nrow(Ytest_levels)-1.
  # The logic is exactly as in "script two":
  #   - Keep track of the last p actual differences in a buffer
  #   - Start from the last training level as "current_level"
  #   - For each step i=1..Ttest:
  #       (i)   Build 1 x (d*p) from the p most recent actual differences
  #       (ii)  Predict the *difference*
  #       (iii) Forecasted level = current_level + predicted_diff
  #       (iv)  Then update "current_level" to the *actual* next level from Ytest_levels
  #             so that subsequent steps use the actual difference
  #
  d <- ncol(Ytrain_diff)
  Ttest <- nrow(Ytest_levels) - 1  # if you have 4 test diffs => 5 test level rows => Ttest=4
  preds <- matrix(NA_real_, nrow=Ttest, ncol=d)
  colnames(preds) <- colnames(Ytrain_levels)

  # The last p diffs from training
  diff_history <- tail(Ytrain_diff, p)  # shape p x d
  # The current "actual" level is last row of the training set
  current_level <- tail(Ytrain_levels, 1)  # shape 1 x d

  for(i in seq_len(Ttest)) {
    # (i) Build the 1 x (d*p) lag vector from diff_history, with the most recent diff first
    #     i.e. diff_history[p, ] is most recent
    lag_vec <- as.vector(t(diff_history[p:1, , drop=FALSE]))  # flatten row by row
    lag_vec <- matrix(lag_vec, nrow=1) # shape: 1 x (d*p)

    # (ii) Predict the next difference
    pred_diff <- lag_vec %*% t(B)   # shape 1 x d

    # (iii) Forecast level = current_level + predicted diff
    pred_level <- current_level + pred_diff
    preds[i, ] <- pred_level

    # (iv) Update to the *actual* next level from Ytest_levels
    #      So that the next step uses the real new difference
    actual_next_level <- Ytest_levels[i+1, , drop=FALSE]
    new_diff <- actual_next_level - current_level

    # Shift the diff_history up by 1, then add the new_diff
    if(p>1) {
      diff_history <- rbind(diff_history[-1, , drop=FALSE], new_diff)
    } else {
      diff_history <- new_diff
    }
    current_level <- actual_next_level
  }
  preds
}

# (Optional) RMSE / MAPE helpers
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm=TRUE))
}
mape <- function(actual, predicted, eps=1e-8) {
  # skip near-zero actual
  idx <- abs(actual) > eps
  if(!any(idx)) return(NA_real_)
  100 * mean(abs((actual[idx] - predicted[idx]) / actual[idx]))
}


# -------------------------
# 5) Choose a lag order p
# -------------------------
p <- 10

# Make design matrix from Ytrain_diff
train_data <- make_VAR_design_p(Ytrain_diff, p)
X_train <- train_data$X  # shape: (Ttrain_diff - p) x (d*p)
Y_train <- train_data$Y  # shape: (Ttrain_diff - p) x d

cat("X_train shape:", dim(X_train), "\n")
cat("Y_train shape:", dim(Y_train), "\n")

# -------------------------
# 6) Fit each method
# -------------------------
## (a) Ridge (frequentist)
lambda_ridge <- 0.1
ridge_fit <- glmnet(
  x = X_train,
  y = Y_train,
  alpha = 0,           # alpha=0 => ridge
  family = "mgaussian",
  lambda = lambda_ridge,
  intercept = FALSE
)

# Extract coefs => (d x d*p)
coef_ridge <- coef(ridge_fit, s=lambda_ridge)
Bhat_ridge <- matrix(NA, nrow=d, ncol=d*p)
for (j in 1:d) {
  Bhat_ridge[j, ] <- as.numeric(coef_ridge[[j]])[2:(d*p + 1)]
}

## (b) Nonparametric Shrinkage (frequentist)
fit_ns <- VARshrink(
  y    = Ytrain_diff,
  p    = p,
  type = "none",  # no intercept in the differences
  method="ns"
)
Bhat_ns <- matrix(NA, nrow=d, ncol=d*p)
for (j in 1:d) {
  Bhat_ns[j, ] <- fit_ns$varresult[[j]]$coefficients
}

## (c) Normal prior (Stan)
stan_data_normal <- list(
  T           = nrow(X_train),  # number of training rows for X
  d           = d,
  p           = p,
  X           = X_train,
  Y           = Y_train,
  prior_scale = 1.0
)
fit_normal <- stan(
  file   = "helicon/stan/var_normal.stan",
  data   = stan_data_normal,
  iter   = 2000,
  warmup = 500,
  chains = 2,
  seed   = 123
)
post_normal <- rstan::extract(fit_normal, "B")  # shape: draws x d x (d*p)
Bhat_normal <- apply(post_normal$B, c(2,3), mean)

## (d) Lasso prior (Stan)
stan_data_lasso <- list(
  T      = nrow(X_train),
  d      = d,
  p      = p,
  X      = X_train,
  Y      = Y_train,
  lambda = 1.0
)
fit_lasso <- stan(
  file   = "helicon/stan/var_lasso.stan",
  data   = stan_data_lasso,
  iter   = 2000,
  warmup = 500,
  chains = 2,
  seed   = 123
)
post_lasso <- rstan::extract(fit_lasso, "B")
Bhat_lasso <- apply(post_lasso$B, c(2,3), mean)

## (e) Horseshoe prior (Stan)
stan_data_hs <- list(
  T = nrow(X_train),
  d = d,
  p = p,
  X = X_train,
  Y = Y_train
)
fit_hs <- stan(
  file   = "helicon/stan/var_horseshoe.stan",
  data   = stan_data_hs,
  iter   = 2000,
  warmup = 500,
  chains = 2,
  seed   = 123
)
post_hs <- rstan::extract(fit_hs, "B")
Bhat_hs <- apply(post_hs$B, c(2,3), mean)

# -------------------------
# 7) Produce rolling 1-step-ahead forecasts on levels
# -------------------------
pred_ridge <- one_step_ahead_forecast(Bhat_ridge, Ytrain_diff, Ytrain_levels, Ytest_levels, p)
pred_ns    <- one_step_ahead_forecast(Bhat_ns,    Ytrain_diff, Ytrain_levels, Ytest_levels, p)
pred_norm  <- one_step_ahead_forecast(Bhat_normal,Ytrain_diff, Ytrain_levels, Ytest_levels, p)
pred_lasso <- one_step_ahead_forecast(Bhat_lasso, Ytrain_diff, Ytrain_levels, Ytest_levels, p)
pred_hs    <- one_step_ahead_forecast(Bhat_hs,    Ytrain_diff, Ytrain_levels, Ytest_levels, p)

# The "actual" test-level rows that match these 1-step forecasts
# If Ytest_levels has 5 rows, the first row is used for initialization,
# so the final 4 rows are the actual next-step levels:
actual_test <- Ytest_levels[2:nrow(Ytest_levels), , drop=FALSE]  # shape (4 x d)
colnames(actual_test) <- colnames(Y_levels)

# -------------------------
# 8) Compute RMSE / MAPE
# -------------------------
methods <- c("Ridge","NS","Normal","Lasso","Horseshoe")
pred_list <- list(pred_ridge, pred_ns, pred_norm, pred_lasso, pred_hs)

rmse_vals <- sapply(pred_list, function(P) rmse(actual_test, P))
mape_vals <- sapply(pred_list, function(P) mape(actual_test, P))

results_df <- data.frame(
  Method = methods,
  RMSE   = rmse_vals,
  MAPE   = mape_vals
)
results_df

# Optionally, per-variable errors:
per_var <- lapply(pred_list, function(P) {
  data.frame(
    var=colnames(P),
    RMSE=sapply(seq_len(d), function(j) rmse(actual_test[,j], P[,j])),
    MAPE=sapply(seq_len(d), function(j) mape(actual_test[,j], P[,j]))
  )
})
names(per_var) <- methods

# Example: Show per-variable error for Horseshoe
per_var$Horseshoe

# -------------------------
# 9) Done!
# -------------------------
cat("\nRolling 1-step-ahead forecast comparison:\n")
print(results_df)










# -- Step A: Build data frames of predicted and actual values in "long" format
library(dplyr)
library(tidyr)
library(ggplot2)

# 1) Actuals
df_actual <- data.frame(
  HoldoutIndex = 1:nrow(actual_test),  # 1..4
  e    = actual_test[, "e"],
  prod = actual_test[, "prod"],
  rw   = actual_test[, "rw"],
  U    = actual_test[, "U"]
)
df_actual_long <- df_actual %>%
  pivot_longer(cols = -HoldoutIndex, names_to = "variable", values_to = "Value") %>%
  mutate(Method = "Actual")

# 2) Horseshoe
df_hs <- data.frame(
  HoldoutIndex = 1:nrow(pred_hs),
  e    = pred_hs[, "e"],
  prod = pred_hs[, "prod"],
  rw   = pred_hs[, "rw"],
  U    = pred_hs[, "U"]
)
df_hs_long <- df_hs %>%
  pivot_longer(cols = -HoldoutIndex, names_to = "variable", values_to = "Value") %>%
  mutate(Method = "Horseshoe")

# 3) Lasso
df_lasso <- data.frame(
  HoldoutIndex = 1:nrow(pred_lasso),
  e    = pred_lasso[, "e"],
  prod = pred_lasso[, "prod"],
  rw   = pred_lasso[, "rw"],
  U    = pred_lasso[, "U"]
)
df_lasso_long <- df_lasso %>%
  pivot_longer(cols = -HoldoutIndex, names_to = "variable", values_to = "Value") %>%
  mutate(Method = "Lasso")

# 4) NonparamShrink
df_ns <- data.frame(
  HoldoutIndex = 1:nrow(pred_ns),
  e    = pred_ns[, "e"],
  prod = pred_ns[, "prod"],
  rw   = pred_ns[, "rw"],
  U    = pred_ns[, "U"]
)
df_ns_long <- df_ns %>%
  pivot_longer(cols = -HoldoutIndex, names_to = "variable", values_to = "Value") %>%
  mutate(Method = "ns")

# 5) Normal
df_normal <- data.frame(
  HoldoutIndex = 1:nrow(pred_norm),
  e    = pred_norm[, "e"],
  prod = pred_norm[, "prod"],
  rw   = pred_norm[, "rw"],
  U    = pred_norm[, "U"]
)
df_normal_long <- df_normal %>%
  pivot_longer(cols = -HoldoutIndex, names_to = "variable", values_to = "Value") %>%
  mutate(Method = "Normal")

# 6) Ridge
df_ridge <- data.frame(
  HoldoutIndex = 1:nrow(pred_ridge),
  e    = pred_ridge[, "e"],
  prod = pred_ridge[, "prod"],
  rw   = pred_ridge[, "rw"],
  U    = pred_ridge[, "U"]
)
df_ridge_long <- df_ridge %>%
  pivot_longer(cols = -HoldoutIndex, names_to = "variable", values_to = "Value") %>%
  mutate(Method = "Ridge")

# -- Combine all into one data frame
df_plot <- bind_rows(
  df_actual_long,
  df_hs_long,
  df_lasso_long,
  df_normal_long,
  df_ns_long,
  df_ridge_long
)

# Optional: Control factor levels so that "Actual" is first, etc.
df_plot$Method <- factor(df_plot$Method,
                         levels = c("Actual","Horseshoe","Lasso","Normal","ns","Ridge"))

# -- Step B: Make the plot
ggplot(df_plot, aes(x = HoldoutIndex, y = Value, color = Method)) +
  geom_line(size = 1) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(
    title = "Forecasts and Actuals on Test Set",
    x = "Holdout Index",
    y = "Value"
  ) +
  # Match earlier figure color scheme:
  scale_color_manual(values = c(
    "Actual"         = "black",
    "Horseshoe"      = "red",
    "Lasso"          = "yellow",
    "Normal"         = "green",
    "ns"             = "blue",  # or "NonparamShrink" or "ns" as needed
    "Ridge"          = "magenta"
  )) +scale_linetype_manual(values = c(
    "Actual"    = "dotted",
    "Horseshoe" = "solid",
    "Lasso"     = "solid",
    "Normal"    = "solid",
    "ns"        = "solid",
    "Ridge"     = "solid"
  )) +
  theme_bw(base_size = 14)




ggplot(df_plot,
       aes(x = HoldoutIndex,
           y = Value,
           color = Method,
           linetype = Method)) +  # <--- Add this mapping
  geom_line(size = 1) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Forecasts and Actuals on Test Set",
       x = "Holdout Index",
       y = "Value") +
  scale_color_manual(values = c(
    "Actual"         = "black",
    "Horseshoe"      = "red",
    "Lasso"          = "#B79F00",
    "Normal"         = "#00BA38",
    "ns"            = "#619CFF",
    "Ridge"          = "#C77CFF"
  )) +
  scale_linetype_manual(values = c(
    "Actual"    = "dotted",
    "Horseshoe" = "solid",
    "Lasso"     = "solid",
    "Normal"    = "solid",
    "ns"        = "solid",
    "Ridge"     = "solid"
  )) +
  theme_bw(base_size = 14)


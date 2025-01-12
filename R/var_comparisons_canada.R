################################################################################
# varp_levels_forecast.R
# ---------------------------------------------------------------------------
# Fits a VAR(p) to the "Canada" dataset (differenced data), using:
#   1) Ridge
#   2) Nonparametric Shrinkage
#   3) Normal prior via Stan
#   4) Lasso prior via Stan
#   5) Horseshoe prior via Stan
# Produces 1-step-ahead forecasts on the differenced scale,
# inverts them to the original (levels) scale, and evaluates both RMSE and MAPE.
# Also plots forecast vs. actual for the final test portion.
################################################################################

# -------------------------
# 1) Load required libraries
# -------------------------
library(vars)
library(VARshrink)
library(glmnet)
library(rstan)
library(tidyverse)

# For faster Stan compilation (optional)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# -------------------------
# 2) Load and prepare data
# -------------------------
data("Canada")
# We'll use the first 4 columns (e, prod, rw, U):
Y_levels <- as.matrix(Canada[, 1:4])  # the original "levels" data
colnames(Y_levels) <- c("e", "prod", "rw", "U")

# Create differenced data for fitting:
Y_diff <- diff(Y_levels)

# We'll set up a holdout of the last 4 "differences" for testing
# i.e. fit on first (Tfull_diff - 4) differences, then test on final 4 differences
Tfull_levels <- nrow(Y_levels)
Tfull_diff   <- nrow(Y_diff)

p <- 6  # number of lags
Ttrain_diff <- Tfull_diff - 4
Ytrain_diff <- Y_diff[1:Ttrain_diff, , drop = FALSE]
Ytest_diff  <- Y_diff[(Ttrain_diff + 1):Tfull_diff, , drop = FALSE]

# For evaluation on the *levels* scale:
# If Y_diff[i, ] = Y_levels[i+1, ] - Y_levels[i, ], then
# Ytest_diff[k, ] corresponds to the difference between
#   Y_levels[Ttrain_diff + k + 1, ] and Y_levels[Ttrain_diff + k, ].
# We'll keep the relevant portion of the levels data for final evaluation:
Ytrain_levels <- Y_levels[1:(Ttrain_diff + 1), , drop = FALSE]
Ytest_levels  <- Y_levels[(Ttrain_diff + 1):Tfull_levels, , drop = FALSE]

cat("Number of rows in Ytest_levels =", nrow(Ytest_levels),
    "(should be 5 if we have 4 test differences)\n")

# -------------------------
# 3) Helper functions
# -------------------------

# (a) Build design matrix for a VAR(p) on *differenced* data
make_VAR_design_p <- function(Y, p) {
  # Y: T x d (already differenced)
  # p: number of lags
  # returns:
  #   X: (T - p) x (d*p)
  #   Y_out: (T - p) x d
  T <- nrow(Y)
  d <- ncol(Y)

  X <- matrix(NA_real_, nrow = T - p, ncol = d * p)
  Y_out <- Y[(p + 1):T, , drop = FALSE]
  for (t in (p + 1):T) {
    row_idx <- t - p
    lags <- c()
    for (lag_i in 1:p) {
      lags <- c(lags, Y[t - lag_i, ])
    }
    X[row_idx, ] <- lags
  }
  list(X = X, Y = Y_out)
}

# (b) Compute accuracy metrics (RMSE and MAPE)
#    actual, predicted: T x d matrices
get_accuracy_metrics <- function(actual, predicted) {
  # actual, predicted: same dimensions
  # We'll average across *all time points and variables* for a single metric:
  #   RMSE = sqrt( mean( (actual - predicted)^2 ) )
  #   MAPE = 100 * mean( abs( (actual - predicted) / actual ) )
  # The latter ignores cases where 'actual' is zero or very close to zero.

  # Vectorize over all elements:
  diff_vals  <- as.numeric(actual - predicted)
  actual_vals <- as.numeric(actual)

  rmse <- sqrt(mean(diff_vals^2))

  # For MAPE, filter out where actual=0 or near zero if needed
  nonzero_idx <- which(abs(actual_vals) > 1e-8)
  mape <- NA_real_
  if (length(nonzero_idx) > 0) {
    mape <- 100 * mean(abs(diff_vals[nonzero_idx] / actual_vals[nonzero_idx]))
  }

  return(list(RMSE = rmse, MAPE = mape))
}

# (c) Forecast function for differenced data -> returns predictions on *levels*
forecast_VARp_on_levels <- function(B, Ytrain_diff, Ytrain_levels, Ytest_diff, p) {
  # B: (d x d*p) fitted on differenced data
  # Ytrain_diff: (Ttrain_diff x d)
  # Ytrain_levels: (Ttrain_diff+1 x d) => original levels up to same time
  # Ytest_diff: (Ttest_diff x d)
  # p: integer
  #
  # Returns a (Ttest_diff x d) matrix of predicted *levels*.

  d <- ncol(Ytrain_diff)
  Ttest_diff <- nrow(Ytest_diff)
  Y_for_forecast_diff <- rbind(tail(Ytrain_diff, p), Ytest_diff)

  # Prepare storage for predictions on the levels scale
  preds_levels <- matrix(NA_real_, nrow = Ttest_diff, ncol = d)

  # Start from the last known "actual" level in training:
  current_level <- tail(Ytrain_levels, 1)  # 1 x d

  # 1-step prediction in differenced space
  predict_VARp_diff <- function(B, y_lags) {
    # B: d x (d*p)
    # y_lags: 1 x (d*p)
    return(y_lags %*% t(B))  # => 1 x d
  }

  # Rolling forecast in differenced domain, revert to levels:
  for (i in seq_len(Ttest_diff)) {
    row_idx <- p + i - 1
    lags <- c()
    for (lag_j in 0:(p - 1)) {
      lags <- c(lags, Y_for_forecast_diff[row_idx - lag_j, ])
    }
    lags <- matrix(lags, nrow = 1)

    # 1) Forecast difference
    pred_diff <- predict_VARp_diff(B, lags)  # 1 x d

    # 2) Convert to levels
    pred_level <- current_level + pred_diff  # 1 x d

    preds_levels[i, ] <- pred_level
    # 3) Update
    current_level <- pred_level
  }
  return(preds_levels)
}

# -------------------------
# 4) Prepare training design for differenced model-fitting
# -------------------------
train_data <- make_VAR_design_p(Ytrain_diff, p)
X_train <- train_data$X   # (Ttrain_diff - p) x (d*p)
Y_train <- train_data$Y   # (Ttrain_diff - p) x d
d <- ncol(Y_train)
cat("Shape of X_train =", dim(X_train), ", Y_train =", dim(Y_train), "\n")

# We'll also define the test set size:
Ttest_diff <- nrow(Ytest_diff)

# The actual test *levels* we compare against are:
#   Ytest_levels[2:(Ttest_diff + 1), ]
# because the first row in Ytest_levels is effectively
# the "initial condition" for the test segment.

actual_test_levels <- Ytest_levels[2:(Ttest_diff + 1), , drop = FALSE]

# -------------------------
# 5) Fit Models
# -------------------------

#####################
# (a) Ridge (glmnet)
#####################
lambda_ridge <- 0.1  # example fixed lambda
ridge_fit <- glmnet(
  x = X_train,
  y = Y_train,
  alpha = 0,           # alpha=0 => ridge
  family = "mgaussian",
  lambda = lambda_ridge,
  intercept = FALSE
)
coef_ridge <- coef(ridge_fit, s = lambda_ridge)
Bhat_ridge <- matrix(NA, nrow = d, ncol = d*p)
for (j in seq_len(d)) {
  Bhat_ridge[j, ] <- as.numeric(coef_ridge[[j]])[2:(d*p + 1)]
}

############################
# (b) Nonparametric Shrinkage
############################
fit_ns <- VARshrink(
  y    = Y_train,    # differenced
  p    = p,
  type = "none",     # no intercept
  method = "ns"
)
varlist <- fit_ns$varresult  # list of length d
Bhat_ns <- matrix(NA, nrow = d, ncol = d*p)
for (j in seq_len(d)) {
  Bhat_ns[j, ] <- varlist[[j]]$coefficients
}

################################
# (c) Bayesian Normal prior (Stan)
################################
stan_data_normal <- list(
  T = nrow(X_train),
  d = d,
  p = p,
  X = X_train,
  Y = Y_train,
  prior_scale = 1.0
)
fit_normal <- stan(
  file = "stan/var_normal.stan",  # must exist locally
  data = stan_data_normal,
  iter = 2000,
  warmup = 500,
  chains = 2,   # reduce if you want faster
  seed = 123,
  control = list(adapt_delta = 0.9, max_treedepth = 12)
)
post_normal <- rstan::extract(fit_normal, pars = "B")
Bhat_normal <- apply(post_normal$B, c(2,3), mean)  # => d x (d*p)

############################
# (d) Bayesian Lasso prior (Stan)
############################
stan_data_lasso <- list(
  T = nrow(X_train),
  d = d,
  p = p,
  X = X_train,
  Y = Y_train,
  lambda = 1.0
)
fit_lasso <- stan(
  file = "stan/var_lasso.stan",  # must exist locally
  data = stan_data_lasso,
  iter = 2000,
  warmup = 500,
  chains = 2,
  seed = 123,
  control = list(adapt_delta = 0.9, max_treedepth = 12)
)
post_lasso <- rstan::extract(fit_lasso, pars = "B")
Bhat_lasso <- apply(post_lasso$B, c(2,3), mean)

############################
# (e) Bayesian Horseshoe prior (Stan)
############################
stan_data_hs <- list(
  T = nrow(X_train),
  d = d,
  p = p,
  X = X_train,
  Y = Y_train
)
fit_hs <- stan(
  file = "stan/var_horseshoe.stan",  # must exist locally
  data = stan_data_hs,
  iter = 2000,
  warmup = 500,
  chains = 2,
  seed = 123,
  control = list(adapt_delta = 0.9, max_treedepth = 12)
)
post_hs <- rstan::extract(fit_hs, pars = "B")
Bhat_hs <- apply(post_hs$B, c(2,3), mean)

# -------------------------
# 6) Evaluate forecasts on the original (levels) scale
# -------------------------
# We'll create a named list of all Bhat matrices, then iterate:
method_list <- list(
  Ridge           = Bhat_ridge,
  NonparamShrink  = Bhat_ns,
  Normal          = Bhat_normal,
  Lasso           = Bhat_lasso,
  Horseshoe       = Bhat_hs
)

results_list <- list()  # to store metrics, predictions, etc.

for (m in names(method_list)) {
  Bmat <- method_list[[m]]
  # 1) Get predictions on levels
  preds_levels <- forecast_VARp_on_levels(Bmat, Ytrain_diff, Ytrain_levels, Ytest_diff, p)
  # 2) Compute metrics
  metrics <- get_accuracy_metrics(actual_test_levels, preds_levels)
  # 3) Store
  results_list[[m]] <- list(
    preds_levels = preds_levels,
    RMSE         = metrics$RMSE,
    MAPE         = metrics$MAPE
  )
}

# Convert results to a tibble
final_results <- tibble(
  Method = names(results_list),
  RMSE_onLevels = sapply(results_list, function(x) x$RMSE),
  MAPE_onLevels = sapply(results_list, function(x) x$MAPE)
)

cat("\nFinal Accuracy Results on the Original (Levels) Scale:\n")
print(final_results)

# -------------------------
# 7) Plot forecast vs. actual for the test set
# -------------------------
# We want a long data frame: columns = (Time, Method, Value, Variable, Type)
# where Type is "Actual" or "Forecast", etc.

# The test set has Ttest_diff rows of predictions, but Ttest_diff+1 actual rows,
# so we compare row i of predictions to row i+1 of Ytest_levels.
# We'll define a small time index for the test portion: 1..Ttest_diff
test_time <- seq_len(Ttest_diff)
actual_df <- data.frame(
  Time     = test_time,
  Variable = rep(colnames(Y_levels), each = Ttest_diff),
  Value    = as.vector(actual_test_levels),
  Method   = "Actual"
)

# Gather predictions from each method
pred_dfs <- list()
for (m in names(results_list)) {
  preds_mat <- results_list[[m]]$preds_levels  # Ttest_diff x d
  df_m <- data.frame(
    Time     = test_time,
    Variable = rep(colnames(Y_levels), each = Ttest_diff),
    Value    = as.vector(preds_mat),
    Method   = m
  )
  pred_dfs[[m]] <- df_m
}
pred_df <- do.call(rbind, pred_dfs)

# Combine actual and predictions
plot_df <- rbind(actual_df, pred_df)

# Plot: facet by variable, color by method
# We'll give "Actual" a special color or line type
ggplot(plot_df, aes(x = Time, y = Value, color = Method)) +
  geom_line() +
  facet_wrap(~ Variable, scales = "free_y") +
  theme_bw(base_size = 14) +
  labs(
    title = "Forecast vs. Actual on Test Set",
    x = "Holdout Index",
    y = "Value"
  )
# If you want "Actual" in black:
# + scale_color_manual(values = c("Actual"="black", "Ridge"="red", ...))

# -------------------------
# 8) Also produce coefficient distribution plots
# -------------------------
desired_order <- c("Horseshoe", "Lasso", "Normal", "NonparamShrink", "Ridge")

make_coefs_df <- function(B_mat, method_name) {
  # Return a data frame with Method as a factor in the desired order
  data.frame(
    Method = factor(method_name, levels = desired_order),
    CoefID = seq_along(B_mat),
    Value  = as.numeric(B_mat)
  )
}

coefs_ridge   <- make_coefs_df(Bhat_ridge,   "Ridge")
coefs_ns      <- make_coefs_df(Bhat_ns,      "NonparamShrink")
coefs_normal  <- make_coefs_df(Bhat_normal,  "Normal")
coefs_lasso   <- make_coefs_df(Bhat_lasso,   "Lasso")
coefs_hs      <- make_coefs_df(Bhat_hs,      "Horseshoe")

all_coefs <- bind_rows(
  coefs_hs,
  coefs_lasso,
  coefs_normal,
  coefs_ns,
  coefs_ridge
)

ggplot(all_coefs, aes(x = Method, y = Value)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Distribution of Estimated Coefficients by Method",
       y = "Coefficient Value",
       x = "")

# Or a violin + boxplot overlay:
ggplot(all_coefs, aes(x = Method, y = Value)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.2, outlier.shape = NA) +
  coord_flip() +
  labs(title = "Distribution of Estimated Coefficients by Method",
       y = "Coefficient Value", x = "")

cat("\nDone! The script fitted the models, evaluated RMSE & MAPE, and plotted forecasts vs. actual.\n")

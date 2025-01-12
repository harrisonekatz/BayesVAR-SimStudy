################################################################################
# extended_simulation_varma_with_bootstrap.R
# ------------------------------------------------------------------------------
# A single script implementing three "robust" simulation scenarios for a VAR,
# measuring both:
#   (1) forecast RMSE
#   (2) parameter RMSE, coverage, interval length
#
# Fix: We implement a naive bootstrap to get coverage for Ridge, NS,
# since placeholders gave artificially low coverage.
#
# Scenarios:
#   1) 3-variable VAR(1), fit p=4 => Overfit
#   2) 20-variable sparse VAR(1), fit p=1 => correct order
#   3) same 20-variable sparse, fit p=4 => overfit
#
# We store after each scenario:
#   scenario1_final.rds, scenario2_final.rds, scenario3_final.rds
################################################################################

library(glmnet)      # for ridge
library(VARshrink)   # for NS (nonparametric Stein)
library(rstan)       # for Bayesian
library(tidyverse)   # for data manipulation
library(MASS)        # for mvrnorm
library(dplyr)

# Optional speed up
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

################################################################################
# 0) Helper Functions
################################################################################

# (A) Make design matrix for VAR(p)
make_VAR_design_p <- function(Y, p) {
  T <- nrow(Y)
  d <- ncol(Y)
  X <- matrix(NA, nrow=T - p, ncol=d*p)
  Y_out <- Y[(p+1):T, , drop=FALSE]

  for(t in (p+1):T) {
    row_idx <- t - p
    lags <- c()
    for(lag_i in 1:p) {
      lags <- c(lags, Y[t - lag_i, ])
    }
    X[row_idx, ] <- lags
  }
  list(X=X, Y=Y_out)
}

# (B) 1-step-ahead forecast
predict_VARp <- function(B, y_lags) {
  # B: d x (d*p)
  # y_lags: 1 x (d*p)
  y_lags %*% t(B)  # => 1 x d
}

# (C) Forecast RMSE
forecast_VARp_rmse <- function(B, Ytrain, Ytest, p) {
  Y_for_forecast <- rbind(tail(Ytrain, p), Ytest)
  Ttest <- nrow(Ytest)
  rmse_vec <- c()

  for (i in (p + 1):(p + Ttest - 1)) {
    lags <- c()
    for (lag_j in 0:(p-1)) {
      lags <- c(lags, Y_for_forecast[i - lag_j, ])
    }
    lags <- matrix(lags, nrow=1)
    actual <- Y_for_forecast[i+1, , drop=FALSE]
    pred <- predict_VARp(B, lags)  # 1 x d
    rmse_vec <- c(rmse_vec, sqrt(mean((actual - pred)^2)))
  }
  mean(rmse_vec)
}

# (D) Basic function to form intervals from estimate + sd
get_freq_ci <- function(B_est, B_sd, alpha=0.05) {
  zval <- qnorm(1 - alpha/2)
  B_lower <- B_est - zval * B_sd
  B_upper <- B_est + zval * B_sd
  list(lower=B_lower, upper=B_upper)
}

# (E) Parameter metrics
compute_param_metrics <- function(B_est, B_lower, B_upper, B_true) {
  in_interval <- (B_true >= B_lower) & (B_true <= B_upper)
  coverage <- mean(in_interval)
  interval_length <- mean(B_upper - B_lower)
  param_rmse <- sqrt(mean((B_est - B_true)^2))
  list(coverage=coverage, int_length=interval_length, param_rmse=param_rmse)
}

# (F) Simple naive bootstrap for freq stderrs
#    For truly dependent time series, consider block bootstrap or another approach.
bootstrap_freq_sd <- function(X, Y, fit_method=c("ridge","ns"), p=1,
                              B_est, n_boot=50, lambda=0.1) {
  d <- ncol(Y)
  n <- nrow(X)
  Bboot <- array(0, dim=c(n_boot, d, ncol(X)))  # shape: n_boot x d x (d*p)

  for(b in 1:n_boot) {
    idx <- sample(seq_len(n), size=n, replace=TRUE)
    Xb <- X[idx, , drop=FALSE]
    Yb <- Y[idx, , drop=FALSE]

    if (fit_method=="ridge") {
      fit <- glmnet(
        x=Xb, y=Yb,
        alpha=0, lambda=lambda, intercept=FALSE, family="mgaussian"
      )
      c_list <- coef(fit, s=lambda)
      tmpB <- matrix(NA, nrow=d, ncol=ncol(X))
      for(j in 1:d) {
        tmpB[j, ] <- as.numeric(c_list[[j]])[-1]
      }
    } else if (fit_method=="ns") {
      fit2 <- VARshrink(y=Yb, p=p, type="none", method="ns")
      varlist <- fit2$varresult
      tmpB <- matrix(NA, nrow=d, ncol=ncol(X))
      for(j in 1:d) {
        tmpB[j, ] <- varlist[[j]]$coefficients
      }
    } else {
      stop("Unknown fit_method for bootstrap.")
    }
    Bboot[b,,] <- tmpB
  }
  # standard deviation across bootstrap replicates
  B_sd <- apply(Bboot, c(2,3), sd)
  B_sd
}

################################################################################
# 1) Data Generation for 3 scenarios
################################################################################

gen_data_scenario1 <- function(T, burnin=50) {
  # 3 x 3
  A <- matrix(c(
    0.5, 0.1, 0,
    0,   0.4, 0.1,
    0.1, 0,   0.3
  ), 3, 3, byrow=TRUE)
  Sigma <- diag(0.05, 3)
  Y <- matrix(0, nrow=T+burnin, ncol=3)
  Y[1,] <- rnorm(3)

  for(t in 2:(T+burnin)) {
    mean_t <- A %*% Y[t-1,]
    Y[t,] <- mvrnorm(1, mu=mean_t, Sigma=Sigma)
  }
  list(Y=Y[(burnin+1):(T+burnin), ], A_true=A, Sigma_true=Sigma)
}

gen_data_scenario2 <- function(T, burnin=50) {
  d <- 20
  A <- matrix(0, d, d)
  for(i in seq_len(d)) {
    for(j in seq_len(d)) {
      if(runif(1) < 0.3) {
        A[i,j] <- 0
      } else {
        A[i,j] <- runif(1, -0.4, 0.4)
      }
    }
  }
  # stationarize
  ev <- eigen(A)$values
  if(max(Mod(ev)) >= 1) {
    A <- A / (1.1 * max(Mod(ev)))
  }
  Sigma <- diag(0.1, d)

  Y <- matrix(0, nrow=T+burnin, ncol=d)
  Y[1,] <- rnorm(d)
  for(t in 2:(T+burnin)) {
    mean_t <- A %*% Y[t-1,]
    Y[t,] <- mvrnorm(1, mu=mean_t, Sigma=Sigma)
  }
  list(Y=Y[(burnin+1):(T+burnin), ], A_true=A, Sigma_true=Sigma)
}

################################################################################
# 2) Main Simulation
################################################################################

N <- 50
results_list <- list(
  scenario1 = list(),
  scenario2 = list(),
  scenario3 = list()
)

save_after_scenario <- function(obj, fname) {
  saveRDS(obj, file=fname)
  cat("\nSaved results to", fname, "\n")
}

# paths to Stan
stan_file_normal    <- "helicon/stan/var_normal.stan"
stan_file_lasso     <- "helicon/stan/var_lasso.stan"
stan_file_horseshoe <- "helicon/stan/var_horseshoe.stan"


################################################################################
# Scenario 1: (3 var, fit p=4)
################################################################################
cat("\n=== SCENARIO 1: 3 var => fit p=4 ===\n")
set.seed(101)
T_s1 <- 200

for(r in seq_len(N)) {
  gen_out <- gen_data_scenario1(T_s1)
  Yfull <- gen_out$Y
  A_true <- gen_out$A_true  # 3x3
  d <- ncol(Yfull)
  Tfull <- nrow(Yfull)

  Ttest <- 20
  Ttrain <- Tfull - Ttest
  Ytrain <- Yfull[1:Ttrain,]
  Ytest  <- Yfull[(Ttrain+1):Tfull,]

  p_fit <- 4
  train_data <- make_VAR_design_p(Ytrain, p_fit)
  X_train <- train_data$X
  Y_train <- train_data$Y

  # true B => first-lag is A_true, rest zero
  B_true_s1 <- cbind(A_true, matrix(0, nrow=3, ncol=3*(p_fit-1)))

  # ---------- Frequentist: Ridge
  lambda_ridge <- 0.1
  ridge_fit <- glmnet(
    x=X_train, y=Y_train,
    alpha=0, lambda=lambda_ridge, intercept=FALSE, family="mgaussian"
  )
  c_list <- coef(ridge_fit, s=lambda_ridge)
  Bhat_ridge <- matrix(NA, nrow=d, ncol=d*p_fit)
  for(j in 1:d) {
    Bhat_ridge[j, ] <- as.numeric(c_list[[j]])[-1]
  }
  # bootstrap stderrs
  Bsd_ridge <- bootstrap_freq_sd(X_train, Y_train,
                                 fit_method="ridge",
                                 p=p_fit, B_est=Bhat_ridge,
                                 n_boot=30, lambda=lambda_ridge)
  freq_ci_ridge <- get_freq_ci(Bhat_ridge, Bsd_ridge)
  ridge_param <- compute_param_metrics(Bhat_ridge,
                                       freq_ci_ridge$lower,
                                       freq_ci_ridge$upper,
                                       B_true_s1)
  ridge_forecast <- forecast_VARp_rmse(Bhat_ridge, Ytrain, Ytest, p_fit)

  # ---------- Frequentist: NS
  fit_ns <- VARshrink(y=Y_train, p=p_fit, type="none", method="ns")
  varlist_ns <- fit_ns$varresult
  Bhat_ns <- matrix(NA, nrow=d, ncol=d*p_fit)
  for(j in 1:d) {
    Bhat_ns[j, ] <- varlist_ns[[j]]$coefficients
  }
  Bsd_ns <- bootstrap_freq_sd(X_train, Y_train,
                              fit_method="ns",
                              p=p_fit, B_est=Bhat_ns,
                              n_boot=30)
  freq_ci_ns <- get_freq_ci(Bhat_ns, Bsd_ns)
  ns_param <- compute_param_metrics(Bhat_ns,
                                    freq_ci_ns$lower,
                                    freq_ci_ns$upper,
                                    B_true_s1)
  ns_forecast <- forecast_VARp_rmse(Bhat_ns, Ytrain, Ytest, p_fit)

  # ---------- Bayesian: Normal
  standata_norm <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train, prior_scale=1.0
  )
  fit_norm <- stan(
    file=stan_file_normal,
    data=standata_norm,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_norm <- rstan::extract(fit_norm, "B")$B  # shape: draws x d x (d*p)
  Bhat_norm <- apply(post_norm, c(2,3), mean)
  B_lower_norm <- apply(post_norm, c(2,3), quantile, probs=0.025)
  B_upper_norm <- apply(post_norm, c(2,3), quantile, probs=0.975)
  norm_param <- compute_param_metrics(Bhat_norm, B_lower_norm, B_upper_norm, B_true_s1)
  norm_forecast <- forecast_VARp_rmse(Bhat_norm, Ytrain, Ytest, p_fit)

  # ---------- Bayesian: Lasso
  standata_lasso <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train, lambda=1.0
  )
  fit_lasso <- stan(
    file=stan_file_lasso,
    data=standata_lasso,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_lasso <- rstan::extract(fit_lasso, "B")$B
  Bhat_lasso <- apply(post_lasso, c(2,3), mean)
  B_lower_lasso <- apply(post_lasso, c(2,3), quantile, probs=0.025)
  B_upper_lasso <- apply(post_lasso, c(2,3), quantile, probs=0.975)
  lasso_param <- compute_param_metrics(Bhat_lasso, B_lower_lasso, B_upper_lasso, B_true_s1)
  lasso_forecast <- forecast_VARp_rmse(Bhat_lasso, Ytrain, Ytest, p_fit)

  # ---------- Bayesian: Horseshoe
  standata_hs <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train
  )
  fit_hs <- stan(
    file=stan_file_horseshoe,
    data=standata_hs,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_hs <- rstan::extract(fit_hs, "B")$B
  Bhat_hs <- apply(post_hs, c(2,3), mean)
  B_lower_hs <- apply(post_hs, c(2,3), quantile, probs=0.025)
  B_upper_hs <- apply(post_hs, c(2,3), quantile, probs=0.975)
  hs_param <- compute_param_metrics(Bhat_hs, B_lower_hs, B_upper_hs, B_true_s1)
  hs_forecast <- forecast_VARp_rmse(Bhat_hs, Ytrain, Ytest, p_fit)

  # store
  results_list$scenario1[[r]] <- tibble(
    replication = r,
    method = c("ridge","ns","normal","lasso","horseshoe"),
    forecast_rmse = c(ridge_forecast, ns_forecast, norm_forecast,
                      lasso_forecast, hs_forecast),
    param_rmse    = c(ridge_param$param_rmse, ns_param$param_rmse,
                      norm_param$param_rmse, lasso_param$param_rmse,
                      hs_param$param_rmse),
    coverage      = c(ridge_param$coverage, ns_param$coverage,
                      norm_param$coverage, lasso_param$coverage,
                      hs_param$coverage),
    int_length    = c(ridge_param$int_length, ns_param$int_length,
                      norm_param$int_length, lasso_param$int_length,
                      hs_param$int_length)
  )
}

saveRDS(results_list, "scenario1_final.rds")
cat("\nScenario 1 finished.\n")


################################################################################
# Scenario 2: 20 var, p=1
################################################################################
cat("\n=== SCENARIO 2: 20 var => fit p=1 ===\n")
set.seed(202)
T_s2 <- 200

for(r in seq_len(N)) {
  gen_out <- gen_data_scenario2(T_s2)
  Yfull <- gen_out$Y
  A_true <- gen_out$A_true
  d <- ncol(Yfull)
  Tfull <- nrow(Yfull)

  Ttest <- 20
  Ttrain <- Tfull - Ttest
  Ytrain <- Yfull[1:Ttrain,]
  Ytest  <- Yfull[(Ttrain+1):Tfull,]

  p_fit <- 1
  train_data <- make_VAR_design_p(Ytrain, p_fit)
  X_train <- train_data$X
  Y_train <- train_data$Y

  B_true_s2 <- A_true  # just one lag

  # ridge
  lambda_ridge <- 0.1
  ridge_fit <- glmnet(
    x=X_train, y=Y_train,
    alpha=0, lambda=lambda_ridge, intercept=FALSE, family="mgaussian"
  )
  c_list <- coef(ridge_fit, s=lambda_ridge)
  Bhat_ridge <- matrix(NA, nrow=d, ncol=d*p_fit)
  for(j in 1:d) {
    Bhat_ridge[j, ] <- as.numeric(c_list[[j]])[-1]
  }
  Bsd_ridge <- bootstrap_freq_sd(X_train, Y_train, "ridge", p=p_fit,
                                 B_est=Bhat_ridge, n_boot=30, lambda=lambda_ridge)
  freq_ci_ridge <- get_freq_ci(Bhat_ridge, Bsd_ridge)
  ridge_param <- compute_param_metrics(Bhat_ridge, freq_ci_ridge$lower, freq_ci_ridge$upper, B_true_s2)
  ridge_forecast <- forecast_VARp_rmse(Bhat_ridge, Ytrain, Ytest, p_fit)

  # ns
  fit2_ns <- VARshrink(y=Y_train, p=p_fit, type="none", method="ns")
  varlist_ns <- fit2_ns$varresult
  Bhat_ns <- matrix(NA, nrow=d, ncol=d*p_fit)
  for(j in 1:d) {
    Bhat_ns[j, ] <- varlist_ns[[j]]$coefficients
  }
  Bsd_ns <- bootstrap_freq_sd(X_train, Y_train, "ns", p=p_fit,
                              B_est=Bhat_ns, n_boot=30)
  freq_ci_ns <- get_freq_ci(Bhat_ns, Bsd_ns)
  ns_param <- compute_param_metrics(Bhat_ns, freq_ci_ns$lower, freq_ci_ns$upper, B_true_s2)
  ns_forecast <- forecast_VARp_rmse(Bhat_ns, Ytrain, Ytest, p_fit)

  # normal
  standata_norm <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train, prior_scale=1.0
  )
  fit_norm <- stan(
    file=stan_file_normal,
    data=standata_norm,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_norm <- rstan::extract(fit_norm, "B")$B
  Bhat_norm <- apply(post_norm, c(2,3), mean)
  B_lower_norm <- apply(post_norm, c(2,3), quantile, probs=0.025)
  B_upper_norm <- apply(post_norm, c(2,3), quantile, probs=0.975)
  norm_param <- compute_param_metrics(Bhat_norm, B_lower_norm, B_upper_norm, B_true_s2)
  norm_forecast <- forecast_VARp_rmse(Bhat_norm, Ytrain, Ytest, p_fit)

  # lasso
  standata_lasso <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train, lambda=1.0
  )
  fit_lasso <- stan(
    file=stan_file_lasso,
    data=standata_lasso,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_lasso <- rstan::extract(fit_lasso, "B")$B
  Bhat_lasso <- apply(post_lasso, c(2,3), mean)
  B_lower_lasso <- apply(post_lasso, c(2,3), quantile, probs=0.025)
  B_upper_lasso <- apply(post_lasso, c(2,3), quantile, probs=0.975)
  lasso_param <- compute_param_metrics(Bhat_lasso, B_lower_lasso, B_upper_lasso, B_true_s2)
  lasso_forecast <- forecast_VARp_rmse(Bhat_lasso, Ytrain, Ytest, p_fit)

  # horseshoe
  standata_hs <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train
  )
  fit_hs <- stan(
    file=stan_file_horseshoe,
    data=standata_hs,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_hs <- rstan::extract(fit_hs, "B")$B
  Bhat_hs <- apply(post_hs, c(2,3), mean)
  B_lower_hs <- apply(post_hs, c(2,3), quantile, probs=0.025)
  B_upper_hs <- apply(post_hs, c(2,3), quantile, probs=0.975)
  hs_param <- compute_param_metrics(Bhat_hs, B_lower_hs, B_upper_hs, B_true_s2)
  hs_forecast <- forecast_VARp_rmse(Bhat_hs, Ytrain, Ytest, p_fit)

  results_list$scenario2[[r]] <- tibble(
    replication=r,
    method = c("ridge","ns","normal","lasso","horseshoe"),
    forecast_rmse = c(ridge_forecast, ns_forecast, norm_forecast,
                      lasso_forecast, hs_forecast),
    param_rmse    = c(ridge_param$param_rmse, ns_param$param_rmse,
                      norm_param$param_rmse, lasso_param$param_rmse,
                      hs_param$param_rmse),
    coverage      = c(ridge_param$coverage, ns_param$coverage,
                      norm_param$coverage, lasso_param$coverage,
                      hs_param$coverage),
    int_length    = c(ridge_param$int_length, ns_param$int_length,
                      norm_param$int_length, lasso_param$int_length,
                      hs_param$int_length)
  )
}

saveRDS(results_list, "scenario2_final.rds")
cat("\nScenario 2 finished.\n")


################################################################################
# Scenario 3: 20 var, T=200, but fit p=4
################################################################################
cat("\n=== SCENARIO 3: 20 var => fit p=4 (overfit) ===\n")
set.seed(303)
T_s3 <- 200

for(r in seq_len(N)) {
  gen_out <- gen_data_scenario2(T_s3)
  Yfull <- gen_out$Y
  A_true <- gen_out$A_true
  d <- ncol(Yfull)
  Tfull <- nrow(Yfull)

  Ttest <- 20
  Ttrain <- Tfull - Ttest
  Ytrain <- Yfull[1:Ttrain,]
  Ytest  <- Yfull[(Ttrain+1):Tfull,]

  p_fit <- 4
  train_data <- make_VAR_design_p(Ytrain, p_fit)
  X_train <- train_data$X
  Y_train <- train_data$Y

  # true B => first-lag = A_true, rest=0
  B_true_s3 <- cbind(A_true, matrix(0, nrow=d, ncol=d*(p_fit-1)))

  # ridge
  lambda_ridge <- 0.1
  ridge_fit <- glmnet(
    x=X_train, y=Y_train,
    alpha=0, lambda=lambda_ridge, intercept=FALSE, family="mgaussian"
  )
  c_list <- coef(ridge_fit, s=lambda_ridge)
  Bhat_ridge <- matrix(NA, nrow=d, ncol=d*p_fit)
  for(j in 1:d) {
    Bhat_ridge[j, ] <- as.numeric(c_list[[j]])[-1]
  }
  Bsd_ridge <- bootstrap_freq_sd(X_train, Y_train, "ridge", p=p_fit,
                                 B_est=Bhat_ridge, n_boot=30, lambda=lambda_ridge)
  freq_ci_ridge <- get_freq_ci(Bhat_ridge, Bsd_ridge)
  ridge_param <- compute_param_metrics(Bhat_ridge, freq_ci_ridge$lower, freq_ci_ridge$upper, B_true_s3)
  ridge_forecast <- forecast_VARp_rmse(Bhat_ridge, Ytrain, Ytest, p_fit)

  # ns
  fit3_ns <- VARshrink(y=Y_train, p=p_fit, type="none", method="ns")
  varlist_ns <- fit3_ns$varresult
  Bhat_ns <- matrix(NA, nrow=d, ncol=d*p_fit)
  for(j in 1:d) {
    Bhat_ns[j, ] <- varlist_ns[[j]]$coefficients
  }
  Bsd_ns <- bootstrap_freq_sd(X_train, Y_train, "ns", p=p_fit,
                              B_est=Bhat_ns, n_boot=30)
  freq_ci_ns <- get_freq_ci(Bhat_ns, Bsd_ns)
  ns_param <- compute_param_metrics(Bhat_ns, freq_ci_ns$lower, freq_ci_ns$upper, B_true_s3)
  ns_forecast <- forecast_VARp_rmse(Bhat_ns, Ytrain, Ytest, p_fit)

  # normal
  standata_norm <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train, prior_scale=1.0
  )
  fit_norm <- stan(
    file=stan_file_normal,
    data=standata_norm,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_norm <- rstan::extract(fit_norm, "B")$B
  Bhat_norm <- apply(post_norm, c(2,3), mean)
  B_lower_norm <- apply(post_norm, c(2,3), quantile, probs=0.025)
  B_upper_norm <- apply(post_norm, c(2,3), quantile, probs=0.975)
  norm_param <- compute_param_metrics(Bhat_norm, B_lower_norm, B_upper_norm, B_true_s3)
  norm_forecast <- forecast_VARp_rmse(Bhat_norm, Ytrain, Ytest, p_fit)

  # lasso
  standata_lasso <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train, lambda=1.0
  )
  fit_lasso <- stan(
    file=stan_file_lasso,
    data=standata_lasso,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_lasso <- rstan::extract(fit_lasso, "B")$B
  Bhat_lasso <- apply(post_lasso, c(2,3), mean)
  B_lower_lasso <- apply(post_lasso, c(2,3), quantile, probs=0.025)
  B_upper_lasso <- apply(post_lasso, c(2,3), quantile, probs=0.975)
  lasso_param <- compute_param_metrics(Bhat_lasso, B_lower_lasso, B_upper_lasso, B_true_s3)
  lasso_forecast <- forecast_VARp_rmse(Bhat_lasso, Ytrain, Ytest, p_fit)

  # horseshoe
  standata_hs <- list(
    T=nrow(X_train), d=d, p=p_fit,
    X=X_train, Y=Y_train
  )
  fit_hs <- stan(
    file=stan_file_horseshoe,
    data=standata_hs,
    iter=2000, warmup=500, chains=4, seed=123,
    control=list(adapt_delta=0.9, max_treedepth=12)
  )
  post_hs <- rstan::extract(fit_hs, "B")$B
  Bhat_hs <- apply(post_hs, c(2,3), mean)
  B_lower_hs <- apply(post_hs, c(2,3), quantile, probs=0.025)
  B_upper_hs <- apply(post_hs, c(2,3), quantile, probs=0.975)
  hs_param <- compute_param_metrics(Bhat_hs, B_lower_hs, B_upper_hs, B_true_s3)
  hs_forecast <- forecast_VARp_rmse(Bhat_hs, Ytrain, Ytest, p_fit)

  results_list$scenario3[[r]] <- tibble(
    replication=r,
    method = c("ridge","ns","normal","lasso","horseshoe"),
    forecast_rmse = c(ridge_forecast, ns_forecast, norm_forecast,
                      lasso_forecast, hs_forecast),
    param_rmse    = c(ridge_param$param_rmse, ns_param$param_rmse,
                      norm_param$param_rmse, lasso_param$param_rmse,
                      hs_param$param_rmse),
    coverage      = c(ridge_param$coverage, ns_param$coverage,
                      norm_param$coverage, lasso_param$coverage,
                      hs_param$coverage),
    int_length    = c(ridge_param$int_length, ns_param$int_length,
                      norm_param$int_length, lasso_param$int_length,
                      hs_param$int_length)
  )
}

saveRDS(results_list, "scenario3_final.rds")
cat("\nScenario 3 finished.\n")
cat("\nAll scenarios done!\n")

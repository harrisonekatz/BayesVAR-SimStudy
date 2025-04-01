data {
  int<lower=1> T;             // number of training rows, e.g. Ttrain - p
  int<lower=1> d;             // dimension of each Y_t
  int<lower=1> p;             // number of lags
  matrix[T, d*p] X;           // design matrix, each row = [y_{t-1}, ..., y_{t-p}]
  matrix[T, d]   Y;           // response matrix, each row = y_t
  real<lower=0> prior_scale;  // prior std dev for Normal(0, prior_scale^2)
}

parameters {
  // -- Coefficients --
  matrix[d, d*p] B;

  // -- Full covariance (Cholesky-based parameterization) --
  cholesky_factor_corr[d] L_Omega;   // Cholesky factor of correlation matrix
  vector<lower=0>[d] sigma_vec;      // per-dimension scale
}

transformed parameters {
  matrix[d, d] Sigma;
  {
    matrix[d, d] L = diag_pre_multiply(sigma_vec, L_Omega);
    Sigma = L * L';
  }
}

model {
  // === Priors ===

  // Prior on coefficients: Normal(0, prior_scale)
  to_vector(B) ~ normal(0, prior_scale);

  // LKJ prior on the correlation part
  L_Omega ~ lkj_corr_cholesky(2.0);

  // Example prior for each dimension's scale, e.g. half-Cauchy
  sigma_vec ~ cauchy(0, 2.5);

  // === Likelihood ===

  for (t in 1:T) {
    // Convert the row of X into a vector for the linear predictor
    vector[d] mu_t = to_vector(X[t] * B');  // predicted mean (d-dimensional)
    vector[d] y_t  = to_vector(Y[t]);       // observed response (d-dimensional)

    // Multivariate normal with mean mu_t and covariance Sigma
    y_t ~ multi_normal(mu_t, Sigma);
  }
}

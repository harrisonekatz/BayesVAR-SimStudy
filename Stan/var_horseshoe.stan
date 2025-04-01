data {
  int<lower=1> T;        // number of time points
  int<lower=1> d;        // dimension of the response vector
  int<lower=1> p;        // number of lags * dimension for regressors
  matrix[T, d*p] X;      // design matrix (T rows, each row is 1 x (d*p))
  matrix[T, d]   Y;      // response matrix (T rows, each row is 1 x d)
}

parameters {
  // Raw coefficients to be scaled by local & global shrinkage
  matrix[d, d*p] B_raw;

  // local scales (one per coefficient)
  vector<lower=0>[d * d*p] lambda;
  // global scale
  real<lower=0> tau;

  // --- NEW: We remove 'sigma' and introduce a full covariance structure ---

  // Cholesky factor of the correlation matrix (d x d)
  cholesky_factor_corr[d] L_Omega;
  // Per-dimension scale parameters
  vector<lower=0>[d] sigma_vec;
}

transformed parameters {
  matrix[d, d*p] B;
  matrix[d, d] Sigma;    // The full covariance matrix

  // 1) Construct B from B_raw using global & local shrinkage
  {
    int k = 1;
    for(i in 1:d) {
      for(j in 1:(d*p)) {
        B[i, j] = B_raw[i, j] * lambda[k] * tau;
        k += 1;
      }
    }
  }

  // 2) Construct the covariance matrix Sigma from L_Omega and sigma_vec
  //    We use the common Stan trick: Sigma = diag_pre_multiply(sigma_vec, L_Omega)
  //                                     * diag_pre_multiply(sigma_vec, L_Omega)'
  //    That ensures Sigma is positive definite.
  {
    matrix[d, d] L = diag_pre_multiply(sigma_vec, L_Omega);
    Sigma = L * L';
  }
}

model {
  // === Prior on the shrinkage parameters ===
  // half-Cauchy(0, 1) for each local scale
  lambda ~ cauchy(0, 1);
  // half-Cauchy(0, 1) for global scale
  tau ~ cauchy(0, 1);

  // Standard normal on B_raw
  to_vector(B_raw) ~ normal(0, 1);

  // === Prior on the covariance structure ===
  // This is just an example “shrinkage” prior for the correlation matrix:
  L_Omega ~ lkj_corr_cholesky(2.0);
    // The parameter 2.0 is an example; it implies a moderate prior belief
    // in correlations being near zero. Adjust as you see fit.

  // half-Cauchy or half-normal on each scale parameter
  sigma_vec ~ cauchy(0, 2.5);
    // This is a somewhat wide prior; tweak as desired.

  // === Likelihood ===
  // Each row of Y[t] is a 1 x d row_vector, we cast it to a vector for multi_normal
  // The mean is (X[t] * B'), also a row_vector of length d. We cast it to vector too.
  for(t in 1:T) {
    vector[d] mu_t = to_vector(X[t] * B');
    vector[d] y_t = to_vector(Y[t]);

    y_t ~ multi_normal(mu_t, Sigma);
  }
}

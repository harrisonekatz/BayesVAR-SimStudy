data {
  int<lower=1> T;         // Number of time points
  int<lower=1> d;         // Dimensionality of the response vector
  int<lower=1> p;         // p "blocks" of regressors (could be lags * d, for example)

  // X is T x (d*p). Each row is the predictors that feed into the
  // entire d-dimensional response at time t.
  matrix[T, d*p] X;

  // Y is T x d. Each row Y[t, ] is a 1 x d row vector
  // representing the d-dimensional response at time t.
  matrix[T, d]   Y;

  // global shrinkage parameter for Laplace prior
  real<lower=0> lambda;
}

parameters {
  // B is d x (d*p) => the matrix of regression coefficients for all d equations
  matrix[d, d*p] B;

  // FULL covariance among the d response dimensions:
  // 1) 'L_Omega' is the Cholesky factor of the correlation matrix
  cholesky_factor_corr[d] L_Omega;

  // 2) Per-dimension scale (standard deviations for each dimension)
  vector<lower=0>[d] sigma_vec;
}

transformed parameters {
  // Construct the full covariance matrix Sigma = L * L'
  // where L = diag_pre_multiply(sigma_vec, L_Omega)
  matrix[d, d] Sigma;

  {
    matrix[d, d] L = diag_pre_multiply(sigma_vec, L_Omega);
    Sigma = L * L';
  }
}

model {
  // 1) PRIOR on the coefficients B:
  //    Laplace(0, scale = 1/lambda)
  //    In Stan: double_exponential(mu=0, b = 1/lambda)
  to_vector(B) ~ double_exponential(0, 1/lambda);

  // 2) PRIOR on the correlation and scale parameters:
  //    We use an LKJ prior on the correlation matrix
  //    plus a half-Cauchy/half-Student or similar on sigma_vec
  L_Omega ~ lkj_corr_cholesky(2.0);   // Example: weaker or stronger correlation prior
  sigma_vec ~ cauchy(0, 2.5);         // Example: half-Cauchy(0, 2.5)

  // 3) LIKELIHOOD:
  //    multi_normal for each time point's d-dimensional response
  //    mean = X[t] * B' (vector of length d), covariance = Sigma (d x d)
  for (t in 1:T) {
    vector[d] mu_t = to_vector(X[t] * B'); // predicted mean
    vector[d] y_t  = to_vector(Y[t]);      // actual response

    y_t ~ multi_normal(mu_t, Sigma);
  }
}

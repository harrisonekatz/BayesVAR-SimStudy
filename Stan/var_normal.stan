data {
  int<lower=1> T;             // number of training rows, e.g. Ttrain - p
  int<lower=1> d;             // dimension of each Y_t
  int<lower=1> p;             // number of lags
  matrix[T, d*p] X;           // design matrix, each row = [y_{t-1}, ..., y_{t-p}]
  matrix[T, d]   Y;           // response matrix, each row = y_t
  real<lower=0> prior_scale;  // prior std dev for Normal(0, prior_scale^2)
}

parameters {
  matrix[d, d*p] B;      // coefficient matrix
  real<lower=0> sigma;   // noise scale
}

model {
  // Normal(0, prior_scale) on each entry of B:
  to_vector(B) ~ normal(0, prior_scale);

  // Likelihood:
  for (t in 1:T) {
    // Y[t, ] is 1 x d
    // X[t, ] is 1 x (d*p)
    // B' is (d*p) x d
    Y[t] ~ normal(X[t] * B', sigma);
  }
}

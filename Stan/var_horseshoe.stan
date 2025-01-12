data {
  int<lower=1> T;
  int<lower=1> d;
  int<lower=1> p;
  matrix[T, d*p] X;
  matrix[T, d]   Y;
}

parameters {
  // Raw coefficients to be scaled by local & global shrinkage
  matrix[d, d*p] B_raw;

  // local scale (one per coefficient):
  vector<lower=0>[d * d*p] lambda;
  // global scale:
  real<lower=0> tau;
  // noise scale:
  real<lower=0> sigma;
}

transformed parameters {
  matrix[d, d*p] B;

  {
    // reshape lambda into a matrix (though we can do it row by row)
    // We'll do an element-by-element multiplication
    // B[i, j] = B_raw[i, j] * (lambda_{ij} * tau)
    int k = 1;
    for(i in 1:d) {
      for(j in 1:(d*p)) {
        B[i, j] = B_raw[i, j] * lambda[k] * tau;
        k += 1;
      }
    }
  }
}

model {
  // half-Cauchy(0,1) on each local scale
  lambda ~ cauchy(0, 1);
  // half-Cauchy(0,1) on the global scale
  tau ~ cauchy(0, 1);

  // standard normal on B_raw
  to_vector(B_raw) ~ normal(0, 1);

  // likelihood
  for(t in 1:T) {
    Y[t] ~ normal(X[t] * B', sigma);
  }
}

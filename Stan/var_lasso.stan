data {
  int<lower=1> T;
  int<lower=1> d;
  int<lower=1> p;
  matrix[T, d*p] X;
  matrix[T, d]   Y;
  real<lower=0> lambda;  // global shrinkage parameter for Laplace prior
}

parameters {
  matrix[d, d*p] B;
  real<lower=0> sigma;
}

model {
  // L1 shrinkage = Laplace(0, scale=1/lambda)
  // Stan's double_exponential param => double_exponential(mu, b)
  // with b = 1/lambda => rate=lambda
  to_vector(B) ~ double_exponential(0, 1/lambda);

  // likelihood
  for (t in 1:T) {
    Y[t] ~ normal(X[t] * B', sigma);
  }
}

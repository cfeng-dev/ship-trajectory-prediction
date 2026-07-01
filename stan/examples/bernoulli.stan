/*
 * @file bernoulli.stan
 * @description Bayesian Bernoulli model with a Beta prior.
 * @date Created on: 01.07.2026
 * @author C.Feng
 */

data {
    // Number of observations
    int<lower=0> N;

    // Binary observations: 1 = success, 0 = failure
    array[N] int<lower=0, upper=1> y;
}

parameters {
    // Unknown probability of success
    real<lower=0, upper=1> theta;
}

model {
    // Uniform prior distribution over theta
    theta ~ beta(1, 1);

    // Likelihood
    y ~ bernoulli(theta);
}
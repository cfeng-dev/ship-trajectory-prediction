/*
 * @file test_cmdstan.stan
 * @description Minimal Stan model for testing the CmdStan installation.
 * @date Created on: 01.07.2026
 * @author C.Feng
 */

parameters {
    // Dummy parameter
    real x;
}

model {
    // Standard normal prior for the dummy parameter
    x ~ normal(0, 1);
}
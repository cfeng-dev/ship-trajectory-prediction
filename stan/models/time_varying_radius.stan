// Constant-speed trajectory model with linearly time-varying curvature.
// Positions use local meters, time uses seconds, heading uses radians,
// curvature uses 1/m, and curvature rate uses 1/(m s). Signed curvature is
// modeled directly; radius is derived and may become very large near a line.

functions {
    vector varying_curvature_step(
        real time_previous,
        real time_step,
        real x_previous,
        real y_previous,
        real speed,
        real heading_previous,
        real curvature_initial,
        real curvature_rate,
        int integration_substeps
    ) {
        vector[3] state;
        real x_current = x_previous;
        real y_current = y_previous;
        real heading_current = heading_previous;
        real substep_time = time_step / integration_substeps;

        for (substep in 1:integration_substeps) {
            real substep_start = time_previous
                + (substep - 1) * substep_time;
            real curvature_start = curvature_initial
                + curvature_rate * substep_start;
            real half_substep = 0.5 * substep_time;
            real heading_midpoint = heading_current
                + speed * (
                    curvature_start * half_substep
                    + 0.5 * curvature_rate * square(half_substep)
                );
            real heading_end = heading_current
                + speed * (
                    curvature_start * substep_time
                    + 0.5 * curvature_rate * square(substep_time)
                );
            real integration_scale = speed * substep_time / 6;

            x_current += integration_scale * (
                cos(heading_current)
                + 4 * cos(heading_midpoint)
                + cos(heading_end)
            );
            y_current += integration_scale * (
                sin(heading_current)
                + 4 * sin(heading_midpoint)
                + sin(heading_end)
            );
            heading_current = heading_end;
        }

        state[1] = x_current;
        state[2] = y_current;
        state[3] = heading_current;
        return state;
    }
}

data {
    int<lower=3> N_observed;
    vector[N_observed] time_observed;
    vector[N_observed] x_observed;
    vector[N_observed] y_observed;

    int<lower=1> N_prediction;
    vector[N_prediction] time_prediction;

    real x_initial;
    real y_initial;
    real<lower=0> speed;
    real heading_initial;
    real curvature_prior_mean;
    real<lower=0> curvature_initial_prior_scale;
    real<lower=0> curvature_rate_prior_scale;
    real<lower=0> sigma_prior_scale;
    int<lower=1> integration_substeps;
}

parameters {
    real curvature_initial_raw;
    real curvature_rate_raw;
    real<lower=0.001> sigma;
}

transformed parameters {
    real curvature_initial = curvature_prior_mean
        + curvature_initial_prior_scale * curvature_initial_raw;
    real curvature_rate = curvature_rate_prior_scale * curvature_rate_raw;
    vector[N_observed] curvature_observed;
    vector[N_observed] radius_observed;
    vector[N_observed] heading_mean;
    vector[N_observed] x_mean;
    vector[N_observed] y_mean;

    curvature_observed[1] = curvature_initial;
    radius_observed[1] = 1 / fmax(abs(curvature_initial), 1e-12);
    heading_mean[1] = heading_initial;
    x_mean[1] = x_initial;
    y_mean[1] = y_initial;

    for (n in 2:N_observed) {
        real time_previous = time_observed[n - 1];
        real time_step = time_observed[n] - time_previous;
        vector[3] state = varying_curvature_step(
            time_previous,
            time_step,
            x_mean[n - 1],
            y_mean[n - 1],
            speed,
            heading_mean[n - 1],
            curvature_initial,
            curvature_rate,
            integration_substeps
        );

        curvature_observed[n] = curvature_initial
            + curvature_rate * time_observed[n];
        radius_observed[n] = 1 / fmax(abs(curvature_observed[n]), 1e-12);
        x_mean[n] = state[1];
        y_mean[n] = state[2];
        heading_mean[n] = state[3];
    }
}

model {
    curvature_initial_raw ~ std_normal();
    curvature_rate_raw ~ std_normal();
    sigma ~ normal(0, sigma_prior_scale);

    x_observed ~ normal(x_mean, sigma);
    y_observed ~ normal(y_mean, sigma);
}

generated quantities {
    real radius_initial = 1 / fmax(abs(curvature_initial), 1e-12);
    real curvature_horizon = curvature_initial
        + curvature_rate * time_prediction[N_prediction];
    real radius_horizon = 1 / fmax(abs(curvature_horizon), 1e-12);
    vector[N_prediction] curvature_prediction;
    vector[N_prediction] turn_rate_prediction;
    vector[N_prediction] radius_prediction;
    vector[N_prediction] heading_prediction_mean;
    vector[N_prediction] x_prediction_mean;
    vector[N_prediction] y_prediction_mean;
    vector[N_prediction] x_prediction;
    vector[N_prediction] y_prediction;
    vector[2 * N_observed] log_likelihood;

    real time_previous = time_observed[N_observed];
    // Rebase the forecast at the final measured position while continuing the
    // fitted heading and linearly changing curvature without interruption.
    real x_previous = x_observed[N_observed];
    real y_previous = y_observed[N_observed];
    real heading_previous = heading_mean[N_observed];

    for (n in 1:N_prediction) {
        real time_step = time_prediction[n] - time_previous;
        vector[3] state = varying_curvature_step(
            time_previous,
            time_step,
            x_previous,
            y_previous,
            speed,
            heading_previous,
            curvature_initial,
            curvature_rate,
            integration_substeps
        );

        curvature_prediction[n] = curvature_initial
            + curvature_rate * time_prediction[n];
        turn_rate_prediction[n] = speed * curvature_prediction[n];
        radius_prediction[n] = 1
            / fmax(abs(curvature_prediction[n]), 1e-12);
        x_prediction_mean[n] = state[1];
        y_prediction_mean[n] = state[2];
        heading_prediction_mean[n] = state[3];
        x_prediction[n] = normal_rng(state[1], sigma);
        y_prediction[n] = normal_rng(state[2], sigma);

        time_previous = time_prediction[n];
        x_previous = state[1];
        y_previous = state[2];
        heading_previous = state[3];
    }

    for (n in 1:N_observed) {
        log_likelihood[n] = normal_lpdf(x_observed[n] | x_mean[n], sigma);
        log_likelihood[N_observed + n] = normal_lpdf(
            y_observed[n] | y_mean[n], sigma
        );
    }
}

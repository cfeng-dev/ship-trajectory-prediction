// Constant-turn-rate-and-velocity model in local meter coordinates.
// Time is measured in seconds, speed in m/s, heading in radians, and signed
// turn rate in rad/s (positive = left, negative = right). Turning radius is
// derived from speed and turn rate rather than sampled independently.

functions {
    real stable_sinc(real value) {
        // Keep the constant-turn formula finite and smooth near zero turn rate.
        if (abs(value) < 1e-4) {
            real value_squared = square(value);
            return 1 - value_squared / 6 + square(value_squared) / 120;
        }
        return sin(value) / value;
    }

    vector constant_turn_position(
        real time,
        real x_initial,
        real y_initial,
        real speed,
        real heading_initial,
        real turn_rate
    ) {
        vector[2] position;
        real half_turn = 0.5 * turn_rate * time;
        real distance = speed * time * stable_sinc(half_turn);
        real midpoint_heading = heading_initial + half_turn;

        position[1] = x_initial + distance * cos(midpoint_heading);
        position[2] = y_initial + distance * sin(midpoint_heading);
        return position;
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
    real<lower=0> speed_prior_median;
    real<lower=0> speed_prior_log_sd;
    real heading_prior_mean;
    real<lower=0> heading_prior_scale;
    real<lower=0> turn_rate_prior_scale;
    real<lower=0> sigma_prior_scale;
}

parameters {
    // Broad physical bounds prevent non-finite warmup proposals.
    real<lower=0.001, upper=100> speed;
    real heading_initial;
    real<lower=-0.1, upper=0.1> turn_rate;
    real<lower=0.001> sigma;
}

transformed parameters {
    vector[N_observed] x_mean;
    vector[N_observed] y_mean;

    for (n in 1:N_observed) {
        vector[2] position = constant_turn_position(
            time_observed[n],
            x_initial,
            y_initial,
            speed,
            heading_initial,
            turn_rate
        );
        x_mean[n] = position[1];
        y_mean[n] = position[2];
    }
}

model {
    speed ~ lognormal(log(speed_prior_median), speed_prior_log_sd);
    heading_initial ~ normal(heading_prior_mean, heading_prior_scale);
    turn_rate ~ normal(0, turn_rate_prior_scale);
    sigma ~ normal(0, sigma_prior_scale);

    x_observed ~ normal(x_mean, sigma);
    y_observed ~ normal(y_mean, sigma);
}

generated quantities {
    // Radius is reported only; near-zero turn rate represents straight motion.
    real radius = speed / fmax(abs(turn_rate), 1e-12);
    vector[N_prediction] x_prediction_mean;
    vector[N_prediction] y_prediction_mean;
    vector[N_prediction] x_prediction;
    vector[N_prediction] y_prediction;
    vector[2 * N_observed] log_likelihood;

    for (n in 1:N_prediction) {
        vector[2] position = constant_turn_position(
            time_prediction[n],
            x_initial,
            y_initial,
            speed,
            heading_initial,
            turn_rate
        );
        x_prediction_mean[n] = position[1];
        y_prediction_mean[n] = position[2];
        x_prediction[n] = normal_rng(x_prediction_mean[n], sigma);
        y_prediction[n] = normal_rng(y_prediction_mean[n], sigma);
    }

    for (n in 1:N_observed) {
        log_likelihood[n] = normal_lpdf(x_observed[n] | x_mean[n], sigma);
        log_likelihood[N_observed + n] = normal_lpdf(
            y_observed[n] | y_mean[n], sigma
        );
    }
}

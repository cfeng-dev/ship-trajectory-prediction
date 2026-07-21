// Constant-turn-rate-and-acceleration (CTRA) model in local meter coordinates.
// Time is measured in seconds, speed in m/s, acceleration in m/s^2, heading
// in radians, and signed turn rate in rad/s (positive = left, negative = right).
// Speed changes linearly while turn rate and tangential acceleration stay fixed.

functions {
    real stable_sinc(real value) {
        // Keep the constant-turn contribution smooth near zero turn rate.
        if (abs(value) < 1e-4) {
            real value_squared = square(value);
            return 1 - value_squared / 6 + square(value_squared) / 120;
        }
        return sin(value) / value;
    }

    real stable_acceleration_turn_factor(real value) {
        // Stable form of (sin(value) - value * cos(value)) / value^2.
        if (abs(value) < 1e-3) {
            real value_squared = square(value);
            return value * (
                1.0 / 3.0
                - value_squared / 30.0
                + square(value_squared) / 840.0
            );
        }
        return (sin(value) - value * cos(value)) / square(value);
    }

    vector constant_turn_acceleration_position(
        real time,
        real x_initial,
        real y_initial,
        real speed_initial,
        real acceleration,
        real heading_initial,
        real turn_rate
    ) {
        vector[2] position;
        real half_turn = 0.5 * turn_rate * time;
        real midpoint_heading = heading_initial + half_turn;
        real midpoint_speed = speed_initial + 0.5 * acceleration * time;
        real along_distance = time * midpoint_speed * stable_sinc(half_turn);
        real cross_distance = 0.5 * acceleration * square(time)
            * stable_acceleration_turn_factor(half_turn);

        position[1] = x_initial
            + along_distance * cos(midpoint_heading)
            - cross_distance * sin(midpoint_heading);
        position[2] = y_initial
            + along_distance * sin(midpoint_heading)
            + cross_distance * cos(midpoint_heading);
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
    real<lower=0> time_horizon;

    real x_initial;
    real y_initial;
    real<lower=0> speed_prior_median;
    real<lower=0> speed_prior_log_sd;
    real heading_prior_mean;
    real<lower=0> heading_prior_scale;
    real<lower=0> turn_rate_prior_scale;
    real<lower=0> acceleration_prior_scale;
    real<lower=0> sigma_prior_scale;
}

parameters {
    // Broad physical bounds prevent non-finite warmup proposals.
    real<lower=0.001, upper=100> speed_initial;
    real<lower=-1, upper=1> acceleration;
    real heading_initial;
    real<lower=-0.1, upper=0.1> turn_rate;
    real<lower=0.001> sigma;
}

transformed parameters {
    real speed_horizon = speed_initial + acceleration * time_horizon;
    vector[N_observed] x_mean;
    vector[N_observed] y_mean;

    for (n in 1:N_observed) {
        vector[2] position = constant_turn_acceleration_position(
            time_observed[n],
            x_initial,
            y_initial,
            speed_initial,
            acceleration,
            heading_initial,
            turn_rate
        );
        x_mean[n] = position[1];
        y_mean[n] = position[2];
    }
}

model {
    // Linear speed is monotonic, so a positive horizon speed keeps it positive
    // throughout the complete observation and prediction interval.
    if (speed_horizon <= 0.001) {
        target += negative_infinity();
    }

    speed_initial ~ lognormal(log(speed_prior_median), speed_prior_log_sd);
    acceleration ~ normal(0, acceleration_prior_scale);
    heading_initial ~ normal(heading_prior_mean, heading_prior_scale);
    turn_rate ~ normal(0, turn_rate_prior_scale);
    sigma ~ normal(0, sigma_prior_scale);

    x_observed ~ normal(x_mean, sigma);
    y_observed ~ normal(y_mean, sigma);
}

generated quantities {
    // With changing speed, the instantaneous radius also changes over time.
    real radius_initial = speed_initial / fmax(abs(turn_rate), 1e-12);
    real radius_horizon = speed_horizon / fmax(abs(turn_rate), 1e-12);
    vector[N_prediction] speed_prediction_mean;
    vector[N_prediction] x_prediction_mean;
    vector[N_prediction] y_prediction_mean;
    vector[N_prediction] x_prediction;
    vector[N_prediction] y_prediction;
    vector[2 * N_observed] log_likelihood;

    for (n in 1:N_prediction) {
        vector[2] position = constant_turn_acceleration_position(
            time_prediction[n],
            x_initial,
            y_initial,
            speed_initial,
            acceleration,
            heading_initial,
            turn_rate
        );
        speed_prediction_mean[n] = speed_initial
            + acceleration * time_prediction[n];
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

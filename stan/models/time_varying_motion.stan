// Bayesian state-space model with time-varying acceleration and turn rate.
// Positions use local meters, time uses seconds, speed uses m/s, tangential
// acceleration uses m/s^2, heading uses radians, and turn rate uses rad/s.
// Acceleration and turn-rate states evolve smoothly and decay toward zero.

functions {
    real stable_sinc(real value) {
        if (abs(value) < 1e-4) {
            real value_squared = square(value);
            return 1 - value_squared / 6 + square(value_squared) / 120;
        }
        return sin(value) / value;
    }

    real stable_acceleration_turn_factor(real value) {
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

    vector motion_step(
        real time_step,
        real x_previous,
        real y_previous,
        real speed_previous,
        real acceleration,
        real heading_previous,
        real turn_rate
    ) {
        vector[2] position;
        real half_turn = 0.5 * turn_rate * time_step;
        real midpoint_heading = heading_previous + half_turn;
        real midpoint_speed = speed_previous + 0.5 * acceleration * time_step;
        real along_distance = time_step * midpoint_speed
            * stable_sinc(half_turn);
        real cross_distance = 0.5 * acceleration * square(time_step)
            * stable_acceleration_turn_factor(half_turn);

        position[1] = x_previous
            + along_distance * cos(midpoint_heading)
            - cross_distance * sin(midpoint_heading);
        position[2] = y_previous
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
    vector<lower=0>[N_observed] speed_observed;

    int<lower=1> N_prediction;
    vector[N_prediction] time_prediction;

    real x_initial;
    real y_initial;
    real<lower=0> speed_prior_median;
    real<lower=0> speed_prior_log_sd;
    real heading_prior_mean;
    real<lower=0> heading_prior_scale;
    real<lower=-0.1, upper=0.1> turn_rate_level;

    real<lower=0> acceleration_initial_scale;
    real<lower=0> acceleration_state_scale;
    real<lower=0> acceleration_decay_time;
    real<lower=0> turn_rate_initial_scale;
    real<lower=0> turn_rate_state_scale;
    real<lower=0> turn_rate_decay_time;

    real<lower=0> sigma_position;
    real<lower=0> sigma_speed;
}

parameters {
    real<lower=0.001, upper=100> speed_initial;
    real heading_initial;

    real<lower=-1, upper=1> acceleration_initial;
    vector[N_observed - 2] acceleration_innovation;
    real<lower=-0.1, upper=0.1> turn_rate_initial;
    vector[N_observed - 2] turn_rate_innovation;
}

transformed parameters {
    vector[N_observed - 1] acceleration_state;
    vector[N_observed - 1] turn_rate_state;
    vector[N_observed] speed_state;
    vector[N_observed] heading_state;
    vector[N_observed] x_mean;
    vector[N_observed] y_mean;

    acceleration_state[1] = acceleration_initial;
    turn_rate_state[1] = turn_rate_initial;
    for (n in 2:(N_observed - 1)) {
        real state_time_step = time_observed[n] - time_observed[n - 1];
        real acceleration_decay = exp(
            -state_time_step / acceleration_decay_time
        );
        real turn_rate_decay = exp(-state_time_step / turn_rate_decay_time);
        acceleration_state[n] = acceleration_decay * acceleration_state[n - 1]
            + acceleration_state_scale
            * sqrt(fmax(1e-12, 1 - square(acceleration_decay)))
            * acceleration_innovation[n - 1];
        turn_rate_state[n] = turn_rate_level
            + turn_rate_decay * (turn_rate_state[n - 1] - turn_rate_level)
            + turn_rate_state_scale
            * sqrt(fmax(1e-12, 1 - square(turn_rate_decay)))
            * turn_rate_innovation[n - 1];
    }

    speed_state[1] = speed_initial;
    heading_state[1] = heading_initial;
    x_mean[1] = x_initial;
    y_mean[1] = y_initial;
    for (n in 2:N_observed) {
        real time_step = time_observed[n] - time_observed[n - 1];
        vector[2] position = motion_step(
            time_step,
            x_mean[n - 1],
            y_mean[n - 1],
            speed_state[n - 1],
            acceleration_state[n - 1],
            heading_state[n - 1],
            turn_rate_state[n - 1]
        );

        speed_state[n] = speed_state[n - 1]
            + acceleration_state[n - 1] * time_step;
        heading_state[n] = heading_state[n - 1]
            + turn_rate_state[n - 1] * time_step;
        x_mean[n] = position[1];
        y_mean[n] = position[2];
    }
}

model {
    // Reject trajectories that imply backward or implausibly fast motion.
    if (min(speed_state) <= 0.001 || max(speed_state) > 100) {
        target += negative_infinity();
    }

    speed_initial ~ lognormal(log(speed_prior_median), speed_prior_log_sd);
    heading_initial ~ normal(heading_prior_mean, heading_prior_scale);
    acceleration_initial ~ normal(0, acceleration_initial_scale);
    acceleration_innovation ~ std_normal();
    turn_rate_initial ~ normal(turn_rate_level, turn_rate_initial_scale);
    turn_rate_innovation ~ std_normal();

    x_observed[2:N_observed] ~ normal(x_mean[2:N_observed], sigma_position);
    y_observed[2:N_observed] ~ normal(y_mean[2:N_observed], sigma_position);
    speed_observed ~ normal(speed_state, sigma_speed);
}

generated quantities {
    vector[N_prediction] acceleration_prediction;
    vector[N_prediction] turn_rate_prediction;
    vector[N_prediction] speed_prediction_mean;
    vector[N_prediction] heading_prediction_mean;
    vector[N_prediction] radius_prediction;
    vector[N_prediction] x_prediction_mean;
    vector[N_prediction] y_prediction_mean;
    vector[N_prediction] speed_prediction;
    vector[N_prediction] x_prediction;
    vector[N_prediction] y_prediction;
    vector[3 * N_observed - 2] log_likelihood;

    real last_state_time_step = time_observed[N_observed]
        - time_observed[N_observed - 1];
    real acceleration_current;
    real turn_rate_current;
    real speed_previous = speed_state[N_observed];
    real heading_previous = heading_state[N_observed];
    real x_previous = x_mean[N_observed];
    real y_previous = y_mean[N_observed];
    real time_previous = time_observed[N_observed];

    acceleration_current = fmin(
        1,
        fmax(
            -1,
            normal_rng(
                exp(-last_state_time_step / acceleration_decay_time)
                    * acceleration_state[N_observed - 1],
                acceleration_state_scale * sqrt(
                    fmax(
                        1e-12,
                        1 - square(
                            exp(
                                -last_state_time_step
                                    / acceleration_decay_time
                            )
                        )
                    )
                )
            )
        )
    );
    turn_rate_current = fmin(
        0.1,
        fmax(
            -0.1,
            normal_rng(
                turn_rate_level
                    + exp(-last_state_time_step / turn_rate_decay_time)
                    * (
                        turn_rate_state[N_observed - 1]
                            - turn_rate_level
                    ),
                turn_rate_state_scale * sqrt(
                    fmax(
                        1e-12,
                        1 - square(
                            exp(-last_state_time_step / turn_rate_decay_time)
                        )
                    )
                )
            )
        )
    );

    for (n in 1:N_prediction) {
        real time_step = time_prediction[n] - time_previous;
        real minimum_acceleration = (0.001 - speed_previous) / time_step;
        real acceleration_used = fmax(
            minimum_acceleration,
            acceleration_current
        );
        vector[2] position = motion_step(
            time_step,
            x_previous,
            y_previous,
            speed_previous,
            acceleration_used,
            heading_previous,
            turn_rate_current
        );
        real speed_current = speed_previous + acceleration_used * time_step;
        real heading_current = heading_previous + turn_rate_current * time_step;

        acceleration_prediction[n] = acceleration_used;
        turn_rate_prediction[n] = turn_rate_current;
        speed_prediction_mean[n] = speed_current;
        heading_prediction_mean[n] = heading_current;
        radius_prediction[n] = speed_current
            / fmax(abs(turn_rate_current), 1e-12);
        x_prediction_mean[n] = position[1];
        y_prediction_mean[n] = position[2];
        speed_prediction[n] = fmax(
            0,
            normal_rng(speed_current, sigma_speed)
        );
        x_prediction[n] = normal_rng(position[1], sigma_position);
        y_prediction[n] = normal_rng(position[2], sigma_position);

        speed_previous = speed_current;
        heading_previous = heading_current;
        x_previous = position[1];
        y_previous = position[2];
        time_previous = time_prediction[n];

        if (n < N_prediction) {
            acceleration_current = fmin(
                1,
                fmax(
                    -1,
                    normal_rng(
                        exp(-time_step / acceleration_decay_time)
                            * acceleration_used,
                        acceleration_state_scale * sqrt(
                            fmax(
                                1e-12,
                                1 - square(
                                    exp(-time_step / acceleration_decay_time)
                                )
                            )
                        )
                    )
                )
            );
            turn_rate_current = fmin(
                0.1,
                fmax(
                    -0.1,
                    normal_rng(
                        turn_rate_level
                            + exp(-time_step / turn_rate_decay_time)
                            * (turn_rate_current - turn_rate_level),
                        turn_rate_state_scale * sqrt(
                            fmax(
                                1e-12,
                                1 - square(
                                    exp(-time_step / turn_rate_decay_time)
                                )
                            )
                        )
                    )
                )
            );
        }
    }

    for (n in 2:N_observed) {
        log_likelihood[n - 1] = normal_lpdf(
            x_observed[n] | x_mean[n], sigma_position
        );
        log_likelihood[N_observed - 1 + n - 1] = normal_lpdf(
            y_observed[n] | y_mean[n], sigma_position
        );
    }
    for (n in 1:N_observed) {
        log_likelihood[2 * (N_observed - 1) + n] = normal_lpdf(
            speed_observed[n] | speed_state[n], sigma_speed
        );
    }
}

// Bayesian constant-curvature trajectory model in local meter coordinates.
// Speed and initial heading are fixed from the observed history. Signed
// curvature is inferred directly: positive turns left, negative turns right,
// and values near zero represent nearly straight motion. Radius is derived as
// the inverse curvature magnitude and remains constant within each draw.

functions {
    real stable_sinc(real value) {
        if (abs(value) < 1e-4) {
            real value_squared = square(value);
            return 1 - value_squared / 6 + square(value_squared) / 120;
        }
        return sin(value) / value;
    }

    vector constant_curvature_position(
        real time,
        real x_initial,
        real y_initial,
        real speed,
        real heading_initial,
        real curvature
    ) {
        vector[2] position;
        real turn_rate = speed * curvature;
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
    real<lower=0> speed;
    real heading_initial;

    real<lower=0> curvature_prior_scale;
    real<lower=0> sigma_prior_scale;
}

parameters {
    real curvature;
    real<lower=0.001> sigma;
}

transformed parameters {
    real turn_rate = speed * curvature;
    vector[N_observed] x_mean;
    vector[N_observed] y_mean;

    for (n in 1:N_observed) {
        vector[2] position = constant_curvature_position(
            time_observed[n],
            x_initial,
            y_initial,
            speed,
            heading_initial,
            curvature
        );
        x_mean[n] = position[1];
        y_mean[n] = position[2];
    }
}

model {
    curvature ~ normal(0, curvature_prior_scale);
    sigma ~ normal(0, sigma_prior_scale);

    x_observed ~ normal(x_mean, sigma);
    y_observed ~ normal(y_mean, sigma);
}

generated quantities {
    real radius = 1 / fmax(abs(curvature), 1e-12);
    int turn_direction;
    vector[N_prediction] x_prediction_mean;
    vector[N_prediction] y_prediction_mean;
    vector[N_prediction] x_prediction;
    vector[N_prediction] y_prediction;
    vector[2 * N_observed] log_likelihood;
    real time_prediction_start = time_observed[N_observed];
    real heading_prediction_start = heading_initial
        + turn_rate * time_prediction_start;
    real x_prediction_start = x_observed[N_observed];
    real y_prediction_start = y_observed[N_observed];

    turn_direction = curvature < 0 ? -1 : 1;

    for (n in 1:N_prediction) {
        real elapsed_time = time_prediction[n] - time_prediction_start;
        vector[2] position = constant_curvature_position(
            elapsed_time,
            x_prediction_start,
            y_prediction_start,
            speed,
            heading_prediction_start,
            curvature
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

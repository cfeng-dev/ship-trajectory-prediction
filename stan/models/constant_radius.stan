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
    int<lower=-1, upper=1> turn_direction;

    real<lower=0> radius_prior_median;
    real<lower=0> radius_prior_log_sd;
    real<lower=0> sigma_prior_scale;
}

parameters {
    real<lower=1, upper=1000000> radius;
    real<lower=0.001> sigma;
}

transformed parameters {
    vector[N_observed] x_mean;
    vector[N_observed] y_mean;
    real signed_radius = turn_direction * radius;
    real angular_velocity = speed / signed_radius;

    for (n in 1:N_observed) {
        real heading = heading_initial + angular_velocity * time_observed[n];
        x_mean[n] = x_initial + signed_radius
            * (sin(heading) - sin(heading_initial));
        y_mean[n] = y_initial - signed_radius
            * (cos(heading) - cos(heading_initial));
    }
}

model {
    radius ~ lognormal(log(radius_prior_median), radius_prior_log_sd);
    sigma ~ normal(0, sigma_prior_scale);

    x_observed ~ normal(x_mean, sigma);
    y_observed ~ normal(y_mean, sigma);
}

generated quantities {
    vector[N_prediction] x_prediction_mean;
    vector[N_prediction] y_prediction_mean;
    vector[N_prediction] x_prediction;
    vector[N_prediction] y_prediction;
    vector[2 * N_observed] log_likelihood;

    for (n in 1:N_prediction) {
        real heading = heading_initial + angular_velocity * time_prediction[n];
        x_prediction_mean[n] = x_initial + signed_radius
            * (sin(heading) - sin(heading_initial));
        y_prediction_mean[n] = y_initial - signed_radius
            * (cos(heading) - cos(heading_initial));
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

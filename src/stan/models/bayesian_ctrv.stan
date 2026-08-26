functions {
  vector ctrv_position(
      real dt,
      real x,
      real y,
      real speed,
      real heading,
      real turn_rate) {
    vector[2] position;

    if (abs(turn_rate) > 1e-6) {
      position[1] = x + speed / turn_rate
          * (sin(heading + turn_rate * dt) - sin(heading));
      position[2] = y + speed / turn_rate
          * (-cos(heading + turn_rate * dt) + cos(heading));
    } else {
      position[1] = x + speed * dt * cos(heading);
      position[2] = y + speed * dt * sin(heading);
    }
    return position;
  }
}

data {
  int<lower=3> N_history;
  vector[N_history] time_observed;
  vector[N_history] x_observed;
  vector[N_history] y_observed;
  real<lower=1e-6> sigma_position_observation_prior_rate;

  int<lower=1> N_prediction;
  vector[N_prediction] time_prediction;

  real<lower=0> speed_prior_scale;
  real<lower=0> turn_rate_prior_scale;
}

transformed data {
  for (n in 2:N_history) {
    if (time_observed[n] <= time_observed[n - 1]) {
      reject("time_observed must be strictly increasing");
    }
  }
  if (time_prediction[1] <= time_observed[N_history]) {
    reject("time_prediction must start after the observations");
  }
  for (n in 2:N_prediction) {
    if (time_prediction[n] <= time_prediction[n - 1]) {
      reject("time_prediction must be strictly increasing");
    }
  }
}

parameters {
  real<lower=0> speed;
  real<lower=-pi(), upper=pi()> heading_initial;
  real<multiplier=turn_rate_prior_scale> turn_rate;
  real<lower=1e-6> sigma_position_observation;
}

transformed parameters {
  vector[N_history] x_model;
  vector[N_history] y_model;
  vector[N_history] heading_model;

  // The first local history position is a fixed observed anchor.
  x_model[1] = x_observed[1];
  y_model[1] = y_observed[1];
  heading_model[1] = heading_initial;

  for (n in 2:N_history) {
    real dt = time_observed[n] - time_observed[n - 1];
    vector[2] position = ctrv_position(
        dt,
        x_model[n - 1],
        y_model[n - 1],
        speed,
        heading_model[n - 1],
        turn_rate);
    x_model[n] = position[1];
    y_model[n] = position[2];
    heading_model[n] = heading_model[n - 1] + turn_rate * dt;
  }
}

model {
  speed ~ normal(0, speed_prior_scale);
  heading_initial ~ uniform(-pi(), pi());
  turn_rate ~ normal(0, turn_rate_prior_scale);
  sigma_position_observation ~ exponential(
      sigma_position_observation_prior_rate);

  // The fixed anchor is excluded; inference starts at history position two.
  for (n in 2:N_history) {
    x_observed[n] ~ normal(x_model[n], sigma_position_observation);
    y_observed[n] ~ normal(y_model[n], sigma_position_observation);
  }
}

generated quantities {
  vector[N_prediction] x_prediction;
  vector[N_prediction] y_prediction;
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;
  vector[2 * (N_history - 1)] log_likelihood;

  real x_previous = x_model[N_history];
  real y_previous = y_model[N_history];
  real heading_previous = heading_model[N_history];
  real time_previous = time_observed[N_history];

  for (n in 2:N_history) {
    int likelihood_index = n - 1;
    log_likelihood[likelihood_index] = normal_lpdf(
        x_observed[n] | x_model[n], sigma_position_observation);
    log_likelihood[N_history - 1 + likelihood_index] = normal_lpdf(
        y_observed[n] | y_model[n], sigma_position_observation);
  }

  for (n in 1:N_prediction) {
    real dt = time_prediction[n] - time_previous;
    vector[2] position = ctrv_position(
        dt,
        x_previous,
        y_previous,
        speed,
        heading_previous,
        turn_rate);

    // Model positions vary only through posterior draws of three kinematic parameters.
    x_prediction[n] = position[1];
    y_prediction[n] = position[2];

    // Future sensor observations additionally include inferred measurement noise.
    x_observation_prediction[n] = normal_rng(
        x_prediction[n], sigma_position_observation);
    y_observation_prediction[n] = normal_rng(
        y_prediction[n], sigma_position_observation);

    x_previous = x_prediction[n];
    y_previous = y_prediction[n];
    heading_previous += turn_rate * dt;
    time_previous = time_prediction[n];
  }
}

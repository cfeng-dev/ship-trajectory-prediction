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
      position[1] = x
                    + speed / turn_rate
                      * (sin(heading + turn_rate * dt) - sin(heading));
      position[2] = y
                    + speed / turn_rate
                      * (-cos(heading + turn_rate * dt) + cos(heading));
    } else {
      position[1] = x + speed * dt * cos(heading);
      position[2] = y + speed * dt * sin(heading);
    }

    return position;
  }
}

data {
  int<lower=2> N_observed;
  vector[N_observed] time_observed;
  vector[N_observed] x_observed;
  vector[N_observed] y_observed;
  vector<lower=0>[N_observed] speed_observed;

  int<lower=1> N_prediction;
  vector[N_prediction] time_prediction;

  real x_initial_prior_mean;
  real y_initial_prior_mean;
  real<lower=0> position_initial_prior_scale;
  real<lower=0> speed_initial_prior_mean;
  real<lower=0> speed_initial_prior_scale;
  real heading_initial_prior_mean;
  real<lower=0> heading_initial_prior_scale;
  real turn_rate_initial_prior_mean;
  real<lower=0> turn_rate_state_prior_scale;
  real<lower=0> turn_rate_limit;

  real<lower=0> sigma_position_gps_prior_scale;
  real<lower=0> sigma_speed_gps_prior_scale;
  real<lower=0> sigma_position_process_prior_scale;
  real<lower=0> sigma_speed_process_prior_scale;
  real<lower=0> sigma_turn_rate_process_prior_scale;
}

transformed data {
  for (n in 2:N_observed) {
    if (time_observed[n] <= time_observed[n - 1]) {
      reject("time_observed must be strictly increasing");
    }
  }
  if (time_prediction[1] <= time_observed[N_observed]) {
    reject("time_prediction must start after the observations");
  }
  for (n in 2:N_prediction) {
    if (time_prediction[n] <= time_prediction[n - 1]) {
      reject("time_prediction must be strictly increasing");
    }
  }
}

parameters {
  vector[N_observed] x_state;
  vector[N_observed] y_state;
  vector<lower=0.001, upper=100>[N_observed] speed_state;
  real heading_initial;
  vector<lower=-turn_rate_limit,
         upper=turn_rate_limit>[N_observed] turn_rate_state;

  real<lower=1e-6> sigma_position_gps;
  real<lower=1e-6> sigma_speed_gps;
  real<lower=1e-6> sigma_position_process;
  real<lower=1e-6> sigma_speed_process;
  real<lower=1e-6> sigma_turn_rate_process;
}

transformed parameters {
  vector[N_observed] heading_state;

  heading_state[1] = heading_initial;
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];
    heading_state[n] = heading_state[n - 1] + turn_rate_state[n - 1] * dt;
  }
}

model {
  x_state[1] ~ normal(x_initial_prior_mean, position_initial_prior_scale);
  y_state[1] ~ normal(y_initial_prior_mean, position_initial_prior_scale);
  speed_state[1] ~ normal(speed_initial_prior_mean, speed_initial_prior_scale);
  heading_initial ~ normal(heading_initial_prior_mean,
                           heading_initial_prior_scale);
  // Regularize the complete latent path, not only its first element. This
  // prevents process noise from supporting implausible full rotations while
  // retaining the local random-walk transition below.
  turn_rate_state ~ normal(turn_rate_initial_prior_mean,
                           turn_rate_state_prior_scale);

  sigma_position_gps ~ normal(0, sigma_position_gps_prior_scale);
  sigma_speed_gps ~ normal(0, sigma_speed_gps_prior_scale);
  sigma_position_process ~ normal(0, sigma_position_process_prior_scale);
  sigma_speed_process ~ normal(0, sigma_speed_process_prior_scale);
  sigma_turn_rate_process ~ normal(0, sigma_turn_rate_process_prior_scale);

  // Propagate the latent state through CTRV. GPS positions enter only the
  // observation model below and never serve as transition inputs.
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];
    vector[2] position = ctrv_position(
        dt,
        x_state[n - 1],
        y_state[n - 1],
        speed_state[n - 1],
        heading_state[n - 1],
        turn_rate_state[n - 1]);

    // Diffusion scales grow with sqrt(dt). Therefore each process sigma is
    // expressed per square-root second in the unit of its corresponding state.
    x_state[n] ~ normal(position[1], sigma_position_process * sqrt(dt));
    y_state[n] ~ normal(position[2], sigma_position_process * sqrt(dt));
    speed_state[n] ~ normal(speed_state[n - 1],
                            sigma_speed_process * sqrt(dt));
    turn_rate_state[n] ~ normal(turn_rate_state[n - 1],
                                sigma_turn_rate_process * sqrt(dt));
  }

  // GPS data are noisy observations of the propagated latent states.
  x_observed ~ normal(x_state, sigma_position_gps);
  y_observed ~ normal(y_state, sigma_position_gps);
  speed_observed ~ normal(speed_state, sigma_speed_gps);
}

generated quantities {
  vector[N_prediction] x_prediction_mean;
  vector[N_prediction] y_prediction_mean;
  vector[N_prediction] speed_prediction_mean;
  vector[N_prediction] x_prediction;
  vector[N_prediction] y_prediction;
  vector[N_prediction] speed_prediction;
  vector[N_prediction] heading_prediction;
  vector[N_prediction] turn_rate_prediction;
  vector[3 * N_observed] log_likelihood;
  // Continue future CTRV propagation from the final latent state, not from
  // the final noisy GPS observation.
  real x_previous = x_state[N_observed];
  real y_previous = y_state[N_observed];
  real speed_previous = speed_state[N_observed];
  real heading_previous = heading_state[N_observed];
  real turn_rate_previous = turn_rate_state[N_observed];
  real time_previous = time_observed[N_observed];

  for (n in 1:N_observed) {
    log_likelihood[n] = normal_lpdf(x_observed[n] | x_state[n],
                                    sigma_position_gps);
    log_likelihood[N_observed + n] = normal_lpdf(
        y_observed[n] | y_state[n], sigma_position_gps);
    log_likelihood[2 * N_observed + n] = normal_lpdf(
        speed_observed[n] | speed_state[n], sigma_speed_gps);
  }

  for (n in 1:N_prediction) {
    real dt = time_prediction[n] - time_previous;
    vector[2] position = ctrv_position(
        dt,
        x_previous,
        y_previous,
        speed_previous,
        heading_previous,
        turn_rate_previous);
    real x_current = normal_rng(
        position[1], sigma_position_process * sqrt(dt));
    real y_current = normal_rng(
        position[2], sigma_position_process * sqrt(dt));
    real speed_current = fmin(
        100,
        fmax(0.001,
             normal_rng(speed_previous, sigma_speed_process * sqrt(dt))));
    real heading_current = heading_previous + turn_rate_previous * dt;
    real turn_rate_current = fmin(
        turn_rate_limit,
        fmax(-turn_rate_limit,
             normal_rng(turn_rate_previous,
                        sigma_turn_rate_process * sqrt(dt))));

    x_prediction_mean[n] = x_current;
    y_prediction_mean[n] = y_current;
    speed_prediction_mean[n] = speed_current;
    x_prediction[n] = normal_rng(x_current, sigma_position_gps);
    y_prediction[n] = normal_rng(y_current, sigma_position_gps);
    speed_prediction[n] = fmax(
        0, normal_rng(speed_current, sigma_speed_gps));
    heading_prediction[n] = heading_current;
    turn_rate_prediction[n] = turn_rate_current;

    x_previous = x_current;
    y_previous = y_current;
    speed_previous = speed_current;
    heading_previous = heading_current;
    turn_rate_previous = turn_rate_current;
    time_previous = time_prediction[n];
  }
}

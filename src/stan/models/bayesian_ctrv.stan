functions {
  real wrap_angle(real angle) {
    return atan2(sin(angle), cos(angle));
  }

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
  real<lower=1e-6> sigma_motion_process_prior_rate;
  real<lower=1e-6> sigma_position_observation_prior_rate;
  real<lower=1e-6> sigma_speed_process_prior_rate;
  real<lower=1e-6> sigma_turn_rate_process_prior_rate;
  real<lower=1e-6> process_reference_interval_seconds;
  real<lower=0> speed_state_lower_mps;

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
  vector[N_history] x_true;
  vector[N_history] y_true;
  vector<lower=speed_state_lower_mps>[N_history] speed_state;
  real<lower=-pi(), upper=pi()> heading_initial;
  vector[N_history] turn_rate_state;
  real<lower=1e-6> sigma_motion_process;
  real<lower=1e-6> sigma_position_observation;
  real<lower=1e-6> sigma_speed_process;
  real<lower=1e-6> sigma_turn_rate_process;
}

transformed parameters {
  vector[N_history] heading_state;

  heading_state[1] = heading_initial;
  for (n in 2:N_history) {
    real dt = time_observed[n] - time_observed[n - 1];
    heading_state[n] = wrap_angle(
        heading_state[n - 1] + turn_rate_state[n] * dt);
  }
}

model {
  speed_state[1] ~ normal(0, speed_prior_scale);
  heading_initial ~ uniform(-pi(), pi());
  turn_rate_state[1] ~ normal(0, turn_rate_prior_scale);
  sigma_motion_process ~ exponential(sigma_motion_process_prior_rate);
  sigma_position_observation ~ exponential(
      sigma_position_observation_prior_rate);
  sigma_speed_process ~ exponential(sigma_speed_process_prior_rate);
  sigma_turn_rate_process ~ exponential(sigma_turn_rate_process_prior_rate);

  // Measurement model for every latent history position.
  x_observed ~ normal(x_true, sigma_position_observation);
  y_observed ~ normal(y_true, sigma_position_observation);

  // Dynamic CTRV transition; position process noise remains per transition.
  for (n in 2:N_history) {
    real dt = time_observed[n] - time_observed[n - 1];
    real process_time_scale = sqrt(
        dt / process_reference_interval_seconds);
    real speed_process_scale = sigma_speed_process * process_time_scale;
    real turn_rate_process_scale = sigma_turn_rate_process * process_time_scale;
    vector[2] position = ctrv_position(
        dt,
        x_true[n - 1],
        y_true[n - 1],
        speed_state[n],
        heading_state[n - 1],
        turn_rate_state[n]);

    // Stan uses a positive truncated additive speed transition. The online
    // RBPF instead reflects negative proposals and rotates heading by pi.
    target += normal_lpdf(
        speed_state[n] | speed_state[n - 1], speed_process_scale)
        - normal_lccdf(
            speed_state_lower_mps | speed_state[n - 1], speed_process_scale);
    turn_rate_state[n] ~ normal(
        turn_rate_state[n - 1], turn_rate_process_scale);
    x_true[n] ~ normal(position[1], sigma_motion_process);
    y_true[n] ~ normal(position[2], sigma_motion_process);
  }
}

generated quantities {
  vector[N_prediction] x_prediction;
  vector[N_prediction] y_prediction;
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;
  vector[2 * N_history] log_likelihood;
  real speed_at_origin = speed_state[N_history];
  real heading_at_origin = heading_state[N_history];
  real turn_rate_at_origin = turn_rate_state[N_history];

  real x_previous = x_true[N_history];
  real y_previous = y_true[N_history];
  real speed_previous = speed_at_origin;
  real heading_previous = heading_at_origin;
  real turn_rate_previous = turn_rate_at_origin;
  real time_previous = time_observed[N_history];

  for (n in 1:N_history) {
    log_likelihood[n] = normal_lpdf(
        x_observed[n] | x_true[n], sigma_position_observation);
    log_likelihood[N_history + n] = normal_lpdf(
        y_observed[n] | y_true[n], sigma_position_observation);
  }

  for (n in 1:N_prediction) {
    real dt = time_prediction[n] - time_previous;
    real process_time_scale = sqrt(
        dt / process_reference_interval_seconds);
    real speed_proposal = normal_rng(
        speed_previous, sigma_speed_process * process_time_scale);
    real heading_for_transition = heading_previous;
    vector[2] expected_position;
    turn_rate_previous = normal_rng(
        turn_rate_previous, sigma_turn_rate_process * process_time_scale);

    // Match the RBPF forecast boundary treatment for future speed proposals.
    if (speed_proposal < 0) {
      speed_previous = fmax(-speed_proposal, speed_state_lower_mps);
      heading_for_transition = wrap_angle(heading_previous + pi());
    } else {
      speed_previous = fmax(speed_proposal, speed_state_lower_mps);
    }

    expected_position = ctrv_position(
        dt,
        x_previous,
        y_previous,
        speed_previous,
        heading_for_transition,
        turn_rate_previous);

    // A model prediction is a future latent state including process noise.
    x_prediction[n] = normal_rng(
        expected_position[1], sigma_motion_process);
    y_prediction[n] = normal_rng(
        expected_position[2], sigma_motion_process);

    // Future sensor observations additionally include inferred measurement noise.
    x_observation_prediction[n] = normal_rng(
        x_prediction[n], sigma_position_observation);
    y_observation_prediction[n] = normal_rng(
        y_prediction[n], sigma_position_observation);

    x_previous = x_prediction[n];
    y_previous = y_prediction[n];
    heading_previous = wrap_angle(
        heading_for_transition + turn_rate_previous * dt);
    time_previous = time_prediction[n];
  }
}

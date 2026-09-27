// One-observation posterior-as-prior update for sequential Bayesian CTRV VI.
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
      position[1] = x + speed / turn_rate * (sin(heading + turn_rate * dt) - sin(heading));
      position[2] = y + speed / turn_rate * (-cos(heading + turn_rate * dt) + cos(heading));
    } else {
      position[1] = x + speed * dt * cos(heading);
      position[2] = y + speed * dt * sin(heading);
    }
    return position;
  }
}

data {
  vector[8] carry_mean;
  matrix[8, 8] carry_cholesky;
  real heading_reference;
  real<lower=1e-6> process_interval_seconds;
  real<lower=1e-6> observation_interval_seconds;
  real x_observed;
  real y_observed;
  real<lower=1e-6> process_reference_interval_seconds;
  real<lower=1e-12> minimum_positive_scale;
}

parameters {
  vector[8] previous_standardized;
  real<lower=0> speed_for_interval;
  real turn_rate_for_interval;
}

transformed parameters {
  vector[8] previous_transformed = carry_mean + carry_cholesky * previous_standardized;
  real previous_x = previous_transformed[1];
  real previous_y = previous_transformed[2];
  real previous_speed = fmax(exp(previous_transformed[3]), minimum_positive_scale);
  real previous_heading = wrap_angle(heading_reference + previous_transformed[4]);
  real previous_turn_rate = previous_transformed[5];
  real sigma_position_observation = fmax(exp(previous_transformed[6]), minimum_positive_scale);
  real sigma_speed_process = fmax(exp(previous_transformed[7]), minimum_positive_scale);
  real sigma_turn_rate_process = fmax(exp(previous_transformed[8]), minimum_positive_scale);
  real process_time_scale = sqrt(process_interval_seconds / process_reference_interval_seconds);
  real speed_process_scale = sigma_speed_process * process_time_scale;
  real turn_rate_process_scale = sigma_turn_rate_process * process_time_scale;
  vector[2] position_at_origin = ctrv_position(
      observation_interval_seconds,
      previous_x,
      previous_y,
      speed_for_interval,
      previous_heading,
      turn_rate_for_interval);
  real x_at_origin = position_at_origin[1];
  real y_at_origin = position_at_origin[2];
  real speed_at_origin = speed_for_interval;
  real heading_at_origin = wrap_angle(
      previous_heading + turn_rate_for_interval * observation_interval_seconds);
  real turn_rate_at_origin = turn_rate_for_interval;
}

model {
  previous_standardized ~ std_normal();

  target += log_sum_exp(
      normal_lpdf(speed_for_interval | previous_speed, speed_process_scale),
      normal_lpdf(-speed_for_interval | previous_speed, speed_process_scale));
  turn_rate_for_interval ~ normal(previous_turn_rate, turn_rate_process_scale);

  x_observed ~ normal(x_at_origin, sigma_position_observation);
  y_observed ~ normal(y_at_origin, sigma_position_observation);
}

generated quantities {
  vector[2] log_likelihood;
  log_likelihood[1] = normal_lpdf(
      x_observed | x_at_origin, sigma_position_observation);
  log_likelihood[2] = normal_lpdf(
      y_observed | y_at_origin, sigma_position_observation);
}

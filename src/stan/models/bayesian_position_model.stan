functions {
  matrix time_scaled_motion_matrix(
      real log_displacement_scale_rate,
      real rotation_rate,
      real dt) {
    real displacement_scale = exp(log_displacement_scale_rate * dt);
    real rotation_angle = rotation_rate * dt;
    matrix[2, 2] result;

    result[1, 1] = displacement_scale * cos(rotation_angle);
    result[1, 2] = -displacement_scale * sin(rotation_angle);
    result[2, 1] = displacement_scale * sin(rotation_angle);
    result[2, 2] = displacement_scale * cos(rotation_angle);
    return result;
  }
}

data {
  int<lower=5> N_history;
  vector[N_history] time_observed;
  vector[N_history] x_observed;
  vector[N_history] y_observed;

  int<lower=1> N_prediction;
  vector[N_prediction] time_prediction;

  real<lower=1e-6> position_model_reference_interval_seconds;
  real<lower=0> log_displacement_scale_rate_prior_scale;
  real<lower=0> rotation_rate_prior_scale;
  real<lower=0> sigma_position_observation_prior_rate;
  real<lower=0> sigma_motion_residual_prior_rate;
}

transformed data {
  for (n in 2:N_history) {
    if (time_observed[n] <= time_observed[n - 1]) {
      reject("time_observed must be strictly increasing");
    }
  }
  if (time_prediction[1] <= time_observed[N_history]) {
    reject("time_prediction must start after time_observed");
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

  // Continuous-time-like local motion rates [1/s] and [rad/s].
  real log_displacement_scale_rate;
  real rotation_rate;

  // Displacement residual SD accumulated over the reference interval [m].
  real<lower=1e-6> sigma_motion_residual;
  real<lower=1e-6> sigma_position_observation;
}

transformed parameters {
  real<lower=0> displacement_scale_at_reference = exp(
      log_displacement_scale_rate
      * position_model_reference_interval_seconds);
  real rotation_angle_at_reference =
      rotation_rate * position_model_reference_interval_seconds;
  array[N_history - 1] vector[2] true_displacement;

  for (n in 1:(N_history - 1)) {
    true_displacement[n][1] = x_true[n + 1] - x_true[n];
    true_displacement[n][2] = y_true[n + 1] - y_true[n];
  }
}

model {
  log_displacement_scale_rate ~ normal(
      0, log_displacement_scale_rate_prior_scale);
  rotation_rate ~ normal(0, rotation_rate_prior_scale);
  sigma_position_observation ~ exponential(
      sigma_position_observation_prior_rate);
  sigma_motion_residual ~ exponential(sigma_motion_residual_prior_rate);

  // Position remains the only sensor observation.
  x_observed ~ normal(x_true, sigma_position_observation);
  y_observed ~ normal(y_true, sigma_position_observation);

  for (n in 2:(N_history - 1)) {
    real dt_previous = time_observed[n] - time_observed[n - 1];
    real dt_current = time_observed[n + 1] - time_observed[n];
    matrix[2, 2] displacement_transition =
        (dt_current / dt_previous)
        * time_scaled_motion_matrix(
            log_displacement_scale_rate,
            rotation_rate,
            dt_current);
    real motion_residual_scale = sigma_motion_residual * sqrt(
        dt_current / position_model_reference_interval_seconds);
    matrix[2, 2] motion_residual_cholesky =
        diag_matrix(rep_vector(motion_residual_scale, 2));

    true_displacement[n] ~ multi_normal_cholesky(
        displacement_transition * true_displacement[n - 1],
        motion_residual_cholesky);
  }
}

generated quantities {
  vector[N_prediction] x_model_prediction;
  vector[N_prediction] y_model_prediction;
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;
  vector[2 * N_history] log_likelihood;

  {
    vector[2] model_position;
    vector[2] model_displacement = true_displacement[N_history - 1];
    real previous_time = time_observed[N_history];
    real previous_dt =
        time_observed[N_history] - time_observed[N_history - 1];

    model_position[1] = x_true[N_history];
    model_position[2] = y_true[N_history];

    for (n in 1:N_prediction) {
      real dt = time_prediction[n] - previous_time;
      matrix[2, 2] displacement_transition =
          (dt / previous_dt)
          * time_scaled_motion_matrix(
              log_displacement_scale_rate,
              rotation_rate,
              dt);
      real motion_residual_scale = sigma_motion_residual * sqrt(
          dt / position_model_reference_interval_seconds);
      matrix[2, 2] motion_residual_cholesky =
          diag_matrix(rep_vector(motion_residual_scale, 2));

      model_displacement = multi_normal_cholesky_rng(
          displacement_transition * model_displacement,
          motion_residual_cholesky);
      model_position += model_displacement;
      x_model_prediction[n] = model_position[1];
      y_model_prediction[n] = model_position[2];

      // Observation noise remains per measurement and is not time-scaled.
      x_observation_prediction[n] = normal_rng(
          x_model_prediction[n], sigma_position_observation);
      y_observation_prediction[n] = normal_rng(
          y_model_prediction[n], sigma_position_observation);

      previous_dt = dt;
      previous_time = time_prediction[n];
    }

    for (n in 1:N_history) {
      log_likelihood[n] = normal_lpdf(
          x_observed[n] | x_true[n], sigma_position_observation);
      log_likelihood[N_history + n] = normal_lpdf(
          y_observed[n] | y_true[n], sigma_position_observation);
    }
  }
}

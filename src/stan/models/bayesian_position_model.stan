data {
  // Last K positions from one complete local observation window
  int<lower=5> N_history;
  vector[N_history] x_observed;
  vector[N_history] y_observed;

  // Fixed position-measurement SD per local x/y axis [m]
  real<lower=1e-6> sigma_position_observation;

  // Number of future trajectory positions
  int<lower=1> N_prediction;

  // Weakly informative position-only prior scales
  real<lower=0> log_displacement_scale_prior_scale;
  real<lower=0> rotation_angle_prior_scale;
  real<lower=0> sigma_motion_residual_prior_scale;
}

parameters {
  // Latent true history positions [m]
  vector[N_history] x_true;
  vector[N_history] y_true;

  // Local rotation-scaling dynamics: A = rho * R(rotation_angle)
  real log_displacement_scale;
  real<lower=-pi(), upper=pi()> rotation_angle;

  // Motion-model residual SD per displacement axis [m]
  real<lower=1e-6> sigma_motion_residual;
}

transformed parameters {
  real<lower=0> displacement_scale = exp(log_displacement_scale);
  matrix[2, 2] autoregressive_matrix;
  array[N_history - 1] vector[2] true_displacement;

  autoregressive_matrix[1, 1] =
      displacement_scale * cos(rotation_angle);
  autoregressive_matrix[1, 2] =
      -displacement_scale * sin(rotation_angle);
  autoregressive_matrix[2, 1] =
      displacement_scale * sin(rotation_angle);
  autoregressive_matrix[2, 2] =
      displacement_scale * cos(rotation_angle);

  for (n in 1:(N_history - 1)) {
    true_displacement[n][1] = x_true[n + 1] - x_true[n];
    true_displacement[n][2] = y_true[n + 1] - y_true[n];
  }
}

model {
  matrix[2, 2] motion_residual_cholesky =
      diag_matrix(rep_vector(sigma_motion_residual, 2));

  // Identity-centered motion prior: rho near 1 and rotation near 0
  log_displacement_scale ~ normal(0, log_displacement_scale_prior_scale);
  rotation_angle ~ normal(0, rotation_angle_prior_scale);
  sigma_motion_residual ~ normal(0, sigma_motion_residual_prior_scale);

  // Explicit measurement-error model with known sensor uncertainty
  x_observed ~ normal(x_true, sigma_position_observation);
  y_observed ~ normal(y_true, sigma_position_observation);

  // Conditional local VAR(1) model for latent true displacements
  for (n in 2:(N_history - 1)) {
    true_displacement[n] ~ multi_normal_cholesky(
        autoregressive_matrix * true_displacement[n - 1],
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
    matrix[2, 2] motion_residual_cholesky =
        diag_matrix(rep_vector(sigma_motion_residual, 2));
    vector[2] model_position;
    vector[2] model_displacement = true_displacement[N_history - 1];

    model_position[1] = x_true[N_history];
    model_position[2] = y_true[N_history];

    for (n in 1:N_prediction) {
      // A model prediction is a future latent trajectory draw.
      model_displacement = multi_normal_cholesky_rng(
          autoregressive_matrix * model_displacement,
          motion_residual_cholesky);
      model_position += model_displacement;
      x_model_prediction[n] = model_position[1];
      y_model_prediction[n] = model_position[2];

      // An observation prediction additionally includes sensor noise.
      x_observation_prediction[n] = normal_rng(
          x_model_prediction[n], sigma_position_observation);
      y_observation_prediction[n] = normal_rng(
          y_model_prediction[n], sigma_position_observation);
    }

    for (n in 1:N_history) {
      log_likelihood[n] = normal_lpdf(
          x_observed[n] | x_true[n], sigma_position_observation);
      log_likelihood[N_history + n] = normal_lpdf(
          y_observed[n] | y_true[n], sigma_position_observation);
    }
  }
}

data {
  // Complete local observation window used by the displacement model
  int<lower=3> N_observed;
  vector[N_observed] x_observed;
  vector[N_observed] y_observed;

  // Number of future observation steps
  int<lower=1> N_prediction;

  // Weakly informative position-only prior scales
  real<lower=0> log_displacement_scale_prior_scale;
  real<lower=0> rotation_angle_prior_scale;
  real<lower=0> sigma_displacement_residual_prior_scale;
}

transformed data {
  array[N_observed - 1] vector[2] displacement;

  for (n in 1:(N_observed - 1)) {
    displacement[n][1] = x_observed[n + 1] - x_observed[n];
    displacement[n][2] = y_observed[n + 1] - y_observed[n];
  }
}

parameters {
  // Local rotation-scaling dynamics: A = rho * R(rotation_angle)
  real log_displacement_scale;
  real<lower=-pi(), upper=pi()> rotation_angle;

  // Total residual SD of the observed displacement transition [m]
  real<lower=1e-6> sigma_displacement_residual;
}

transformed parameters {
  real<lower=0> displacement_scale = exp(log_displacement_scale);
  matrix[2, 2] autoregressive_matrix;

  autoregressive_matrix[1, 1] =
      displacement_scale * cos(rotation_angle);
  autoregressive_matrix[1, 2] =
      -displacement_scale * sin(rotation_angle);
  autoregressive_matrix[2, 1] =
      displacement_scale * sin(rotation_angle);
  autoregressive_matrix[2, 2] =
      displacement_scale * cos(rotation_angle);
}

model {
  matrix[2, 2] residual_cholesky =
      diag_matrix(rep_vector(sigma_displacement_residual, 2));

  // Identity-centered motion prior: rho near 1 and rotation near 0
  log_displacement_scale ~ normal(0, log_displacement_scale_prior_scale);
  rotation_angle ~ normal(0, rotation_angle_prior_scale);
  sigma_displacement_residual ~ normal(
      0,
      sigma_displacement_residual_prior_scale);

  // Local VAR(1) likelihood for directly observed displacements
  for (n in 2:(N_observed - 1)) {
    displacement[n] ~ multi_normal_cholesky(
        autoregressive_matrix * displacement[n - 1],
        residual_cholesky);
  }
}

generated quantities {
  vector[N_prediction] x_model_prediction;
  vector[N_prediction] y_model_prediction;
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;
  vector[N_observed - 2] log_likelihood;

  {
    matrix[2, 2] residual_cholesky =
        diag_matrix(rep_vector(sigma_displacement_residual, 2));
    vector[2] model_position;
    vector[2] model_displacement = displacement[N_observed - 1];
    vector[2] observation_position;
    vector[2] observation_displacement = displacement[N_observed - 1];

    model_position[1] = x_observed[N_observed];
    model_position[2] = y_observed[N_observed];
    observation_position = model_position;

    for (n in 1:N_prediction) {
      model_displacement = autoregressive_matrix * model_displacement;
      model_position += model_displacement;
      x_model_prediction[n] = model_position[1];
      y_model_prediction[n] = model_position[2];

      observation_displacement = multi_normal_cholesky_rng(
          autoregressive_matrix * observation_displacement,
          residual_cholesky);
      observation_position += observation_displacement;
      x_observation_prediction[n] = observation_position[1];
      y_observation_prediction[n] = observation_position[2];
    }

    for (n in 2:(N_observed - 1)) {
      log_likelihood[n - 1] = multi_normal_cholesky_lpdf(
          displacement[n] |
          autoregressive_matrix * displacement[n - 1],
          residual_cholesky);
    }
  }
}

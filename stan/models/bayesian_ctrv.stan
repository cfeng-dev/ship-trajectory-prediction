functions {
  // Deterministic CTRV position update
  vector ctrv_position(
      real dt,
      real x,
      real y,
      real speed,
      real heading,
      real turn_rate) {
    vector[2] position;

    if (abs(turn_rate) > 1e-6) {
      // Exact circular-motion equations
      position[1] = x
                    + speed / turn_rate
                      * (sin(heading + turn_rate * dt) - sin(heading));
      position[2] = y
                    + speed / turn_rate
                      * (-cos(heading + turn_rate * dt) + cos(heading));
    } else {
      // Stable straight-line approximation
      position[1] = x + speed * dt * cos(heading);
      position[2] = y + speed * dt * sin(heading);
    }

    return position;
  }
}

data {
  // Noisy position observations used as external-trajectory proxies
  int<lower=2> N_observed;                         // number of observations
  vector[N_observed] time_observed;                // elapsed time [s]
  vector[N_observed] x_observed;                   // local x position [m]
  vector[N_observed] y_observed;                   // local y position [m]

  // Prediction horizon
  int<lower=1> N_prediction;                       // number of future steps
  vector[N_prediction] time_prediction;            // future elapsed time [s]

  // Initial-state priors
  real x_initial_prior_mean;
  real y_initial_prior_mean;
  real<lower=0> position_initial_prior_scale;
  real<lower=0> speed_initial_prior_mean;
  real<lower=0> speed_initial_prior_scale;
  real heading_initial_prior_mean;
  real<lower=0> heading_initial_prior_scale;
  real turn_rate_initial_prior_mean;
  real<lower=0> turn_rate_state_prior_scale;
  real<lower=0> turn_rate_limit;                    // absolute limit [rad/s]

  // Observation- and process-noise prior scales
  real<lower=0> sigma_position_gps_prior_scale;
  real<lower=0> sigma_position_process_prior_scale;
  real<lower=0> sigma_speed_process_prior_scale;
  real<lower=0> sigma_turn_rate_process_prior_scale;
}

transformed data {
  // Require strictly ordered observation and prediction times
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
  // Latent motion states
  vector[N_observed] x_state;
  vector[N_observed] y_state;
  // Zero permits stationary motion. VI initial values must remain inside bounds.
  vector<lower=0, upper=100>[N_observed] speed_state;
  real heading_initial;
  vector<lower=-turn_rate_limit,
         upper=turn_rate_limit>[N_observed] turn_rate_state;

  // Observation noise
  real<lower=1e-6> sigma_position_gps;

  // Process noise per square-root second
  real<lower=1e-6> sigma_position_process;
  real<lower=1e-6> sigma_speed_process;
  real<lower=1e-6> sigma_turn_rate_process;
}

transformed parameters {
  vector[N_observed] heading_state;

  // Deterministic heading propagation
  heading_state[1] = heading_initial;
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];
    heading_state[n] = heading_state[n - 1] + turn_rate_state[n - 1] * dt;
  }
}

model {
  // Initial-state priors
  x_state[1] ~ normal(x_initial_prior_mean, position_initial_prior_scale);
  y_state[1] ~ normal(y_initial_prior_mean, position_initial_prior_scale);
  speed_state[1] ~ normal(speed_initial_prior_mean, speed_initial_prior_scale);
  heading_initial ~ normal(heading_initial_prior_mean,
                           heading_initial_prior_scale);

  // Prior for the complete latent turn-rate path
  turn_rate_state ~ normal(turn_rate_initial_prior_mean,
                           turn_rate_state_prior_scale);

  // Half-normal priors due to positive sigma constraints
  sigma_position_gps ~ normal(0, sigma_position_gps_prior_scale);
  sigma_position_process ~ normal(0, sigma_position_process_prior_scale);
  sigma_speed_process ~ normal(0, sigma_speed_process_prior_scale);
  sigma_turn_rate_process ~ normal(0, sigma_turn_rate_process_prior_scale);

  // Stochastic CTRV transitions between latent states
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];
    vector[2] position = ctrv_position(
        dt,
        x_state[n - 1],
        y_state[n - 1],
        speed_state[n - 1],
        heading_state[n - 1],
        turn_rate_state[n - 1]);

    // Diffusion scaling for variable time intervals
    x_state[n] ~ normal(position[1], sigma_position_process * sqrt(dt));
    y_state[n] ~ normal(position[2], sigma_position_process * sqrt(dt));
    speed_state[n] ~ normal(speed_state[n - 1],
                            sigma_speed_process * sqrt(dt));
    turn_rate_state[n] ~ normal(turn_rate_state[n - 1],
                                sigma_turn_rate_process * sqrt(dt));
  }

  // Position-only observation model: observations differ from latent truth
  x_observed ~ normal(x_state, sigma_position_gps);
  y_observed ~ normal(y_state, sigma_position_gps);
}

generated quantities {
  // Predicted latent states, including process noise
  vector[N_prediction] x_state_prediction;
  vector[N_prediction] y_state_prediction;
  vector[N_prediction] speed_state_prediction;
  vector[N_prediction] heading_state_prediction;
  vector[N_prediction] turn_rate_state_prediction;

  // Posterior predictive noisy position observations
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;

  // Pointwise position-observation log likelihood
  vector[2 * N_observed] log_likelihood;

  // Start forecasting at the final latent state
  real x_previous = x_state[N_observed];
  real y_previous = y_state[N_observed];
  real speed_previous = speed_state[N_observed];
  real heading_previous = heading_state[N_observed];
  real turn_rate_previous = turn_rate_state[N_observed];
  real time_previous = time_observed[N_observed];

  // Log likelihood for model comparison
  for (n in 1:N_observed) {
    log_likelihood[n] = normal_lpdf(x_observed[n] | x_state[n],
                                    sigma_position_gps);
    log_likelihood[N_observed + n] = normal_lpdf(
        y_observed[n] | y_state[n], sigma_position_gps);
  }

  // Posterior predictive trajectory
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
        fmax(0,
             normal_rng(speed_previous, sigma_speed_process * sqrt(dt))));
    real heading_current = heading_previous + turn_rate_previous * dt;
    real turn_rate_current = fmin(
        turn_rate_limit,
        fmax(-turn_rate_limit,
             normal_rng(turn_rate_previous,
                        sigma_turn_rate_process * sqrt(dt))));

    x_state_prediction[n] = x_current;
    y_state_prediction[n] = y_current;
    speed_state_prediction[n] = speed_current;
    heading_state_prediction[n] = heading_current;
    turn_rate_state_prediction[n] = turn_rate_current;
    x_observation_prediction[n] = normal_rng(x_current, sigma_position_gps);
    y_observation_prediction[n] = normal_rng(y_current, sigma_position_gps);

    x_previous = x_current;
    y_previous = y_current;
    speed_previous = speed_current;
    heading_previous = heading_current;
    turn_rate_previous = turn_rate_current;
    time_previous = time_prediction[n];
  }
}

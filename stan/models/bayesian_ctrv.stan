functions {
  /**
   * Compute the conditional mean of the next 2D position under the
   * constant-turn-rate-and-velocity (CTRV) model, including its
   * straight-line CV limit for near-zero turn rate.
   *
   * Speed and turn rate are assumed constant over the interval dt.
   *
   * Used in the model block for latent-state transitions and in
   * generated quantities for posterior predictive forecasting.
   *
   * @param  dt        Time interval [s]
   * @param  x         Current latent x-position [m]
   * @param  y         Current latent y-position [m]
   * @param  speed     Current latent speed [m/s]
   * @param  heading   Current latent heading [rad]
   * @param  turn_rate Current latent turn rate [rad/s]
   *
   * @return Length-2 vector with the conditional mean of next latent position:
   *         position[1] = next latent x-position mean [m],
   *         position[2] = next latent y-position mean [m].
   */
  vector ctrv_position(real dt, real x, real y, real speed, real heading, real turn_rate) {
    vector[2] position;

    if (abs(turn_rate) > 1e-6) {
      // CTRV motion for non-zero turn rate.
      position[1] = x + speed / turn_rate * (sin(heading + turn_rate * dt) - sin(heading));
      position[2] = y + speed / turn_rate * (-cos(heading + turn_rate * dt) + cos(heading));
    } else {
      // Straight-line CV limit of CTRV for near-zero turn rate.
      position[1] = x + speed * dt * cos(heading);
      position[2] = y + speed * dt * sin(heading);
    }

    return position;
  }
}


data {
  // Observed position data (D) used in the likelihood
  int<lower=2> N_observed;              // Number of observations (must be >= 2)
  vector[N_observed] time_observed;     // Observation times [s]
  vector[N_observed] x_observed;        // Observed x-position [m]
  vector[N_observed] y_observed;        // Observed y-position [m]

  // Future time points for posterior predictive forecasting
  int<lower=1> N_prediction;            // Number of prediction steps (must be >= 1)
  vector[N_prediction] time_prediction; // Prediction times [s]

  // Initial-position prior hyperparameters [m]
	real x_initial_prior_mean;
	real y_initial_prior_mean;
	real<lower=0> position_initial_prior_scale;

	// Initial-speed prior hyperparameters [m/s]
	real<lower=0> speed_initial_prior_mean;
	real<lower=0> speed_initial_prior_scale;

	// Fixed terminal motion derived from recent observed positions
	real heading_final;   // Endpoint heading [rad]
	real turn_rate_final; // Local turn rate [rad/s]

	// Turn-rate prior hyperparameters [rad/s]
	real turn_rate_initial_prior_mean;
	real<lower=0> turn_rate_state_prior_scale;

  // Prior scales for observation and process noise
  real<lower=0> sigma_position_gps_prior_scale;         // Observation-noise prior scale [m]
  real<lower=0> sigma_position_process_prior_scale;     // Position-process prior scale [m/sqrt(s)]
  real<lower=0> sigma_speed_process_prior_scale;        // Speed-process prior scale [(m/s)/sqrt(s)]
  real<lower=0> sigma_turn_rate_process_prior_scale;    // Turn-rate-process prior scale [(rad/s)/sqrt(s)]
}


transformed data {
  // Validate temporal consistency before evaluating the state-space model.

  // Observation times must be strictly increasing.
  for (n in 2:N_observed) {
    if (time_observed[n] <= time_observed[n - 1]) {
      reject("time_observed must be strictly increasing");
    }
  }

  // Prediction must start after the final observation.
  if (time_prediction[1] <= time_observed[N_observed]) {
    reject("time_prediction must start after the observations");
  }

  // Prediction times must be strictly increasing.
  for (n in 2:N_prediction) {
    if (time_prediction[n] <= time_prediction[n - 1]) {
      reject("time_prediction must be strictly increasing");
    }
  }
}


parameters {
  // Define the unknown quantities whose posterior distributions are inferred from the data.

  // Latent position states
  vector[N_observed] x_state;  // Latent true x-position [m]
  vector[N_observed] y_state;  // Latent true y-position [m]

  // Latent kinematic states
  vector<lower=0>[N_observed] speed_state;  // Latent non-negative speed [m/s]
  vector[N_observed] turn_rate_state;       // Latent turn rate [rad/s]

  // Unknown observation-noise parameter
  real<lower=1e-6> sigma_position_gps;  // SD between observed and latent positions [m]

  // Unknown process-noise parameters describing deviations from ideal CTRV dynamics
  real<lower=1e-6> sigma_position_process;  // Position-process diffusion scale [m/sqrt(s)]
  real<lower=1e-6> sigma_speed_process;     // Speed-process diffusion scale [(m/s)/sqrt(s)]
  real<lower=1e-6> sigma_turn_rate_process; // Turn-rate-process diffusion scale [(rad/s)/sqrt(s)]
}


transformed parameters {
  // Deterministically derive the heading trajectory from inferred parameters.

  vector[N_observed] heading_state;  // Derived heading at each observation time [rad]

  heading_state[N_observed] = heading_final;

  // Reconstruct earlier headings by integrating the latent turn rate backward.
  for (reverse_index in 1:(N_observed - 1)) {
    int n = N_observed - reverse_index;
    real dt = time_observed[n + 1] - time_observed[n];

    heading_state[n] = heading_state[n + 1] - turn_rate_state[n] * dt;
  }
}


model {
  /*
   * BAYESIAN MODEL
   *
   * The posterior is induced by all priors, state-transition
   * distributions, and likelihood terms defined below:
   *
   *   posterior is proportional to priors * transitions * likelihood.
   */

  // ------------------------------------------------------------------
  // 1. PRIORS FOR THE INITIAL LATENT STATE
  // ------------------------------------------------------------------

  // Initial latent position: p(x_1, y_1)
  x_state[1] ~ normal(x_initial_prior_mean, position_initial_prior_scale);
  y_state[1] ~ normal(y_initial_prior_mean, position_initial_prior_scale);

  // Initial latent speed: p(v_1), constrained to non-negative values
  speed_state[1] ~ normal(speed_initial_prior_mean, speed_initial_prior_scale);

  // ------------------------------------------------------------------
  // 2. PRIOR REGULARIZATION OF THE LATENT TURN-RATE TRAJECTORY
  // ------------------------------------------------------------------

  /*
   * Marginal prior for the latent turn-rate path:
   *
   *   p(turn_rate_1, ..., turn_rate_N)
   *
   * Regularizes all turn-rate states around the specified prior mean.
   */
  turn_rate_state ~ normal(turn_rate_initial_prior_mean, turn_rate_state_prior_scale);


  // ------------------------------------------------------------------
  // 3. PRIORS FOR OBSERVATION- AND PROCESS-NOISE PARAMETERS
  // ------------------------------------------------------------------

  /*
   * Half-normal priors for positive noise parameters:
   *
   *   sigma ~ HalfNormal(prior_scale)
   */
  sigma_position_gps ~ normal(0, sigma_position_gps_prior_scale);
  sigma_position_process ~ normal(0, sigma_position_process_prior_scale);
  sigma_speed_process ~ normal(0, sigma_speed_process_prior_scale);
  sigma_turn_rate_process ~ normal(0, sigma_turn_rate_process_prior_scale);


  // ------------------------------------------------------------------
  // 4. STOCHASTIC STATE-TRANSITION MODEL / PROCESS MODEL
  // ------------------------------------------------------------------

  /*
   * State-transition distributions:
   *
   *   p(state_n | state_{n-1}, process-noise parameters)
   *
   * CTRV provides the conditional mean; process noise allows deviations.
   */
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];

    // Conditional mean of the next position under deterministic CTRV.
    vector[2] position = ctrv_position(
        dt,
        x_state[n - 1],
        y_state[n - 1],
        speed_state[n - 1],
        heading_state[n - 1],
        turn_rate_state[n - 1]);

    // Position transition with diffusion-scaled process noise.
    x_state[n] ~ normal(position[1], sigma_position_process * sqrt(dt));
    y_state[n] ~ normal(position[2], sigma_position_process * sqrt(dt));

    // Gaussian random walk for latent speed.
    speed_state[n] ~ normal(speed_state[n - 1], sigma_speed_process * sqrt(dt));

    // Gaussian random walk for latent turn rate.
    turn_rate_state[n] ~ normal(turn_rate_state[n - 1], sigma_turn_rate_process * sqrt(dt));
  }


  // ------------------------------------------------------------------
  // 5. LIKELIHOOD / POSITION-OBSERVATION MODEL
  // ------------------------------------------------------------------

  /*
   * Position-only likelihood:
   *
   *   p(x_observed, y_observed | x_state, y_state, sigma_position_gps)
   *
   * Observed positions are noisy measurements of the latent positions.
   */
  x_observed ~ normal(x_state, sigma_position_gps);
  y_observed ~ normal(y_state, sigma_position_gps);
}


generated quantities {
  /*
   * POSTERIOR-DERIVED QUANTITIES
   *
   * Generate posterior predictive trajectories and log-likelihood values
   * without affecting posterior inference.
   */

  // Posterior predictive latent states (including process uncertainty)
  vector[N_prediction] x_state_prediction;
  vector[N_prediction] y_state_prediction;
  vector[N_prediction] speed_state_prediction;
  vector[N_prediction] heading_state_prediction;
  vector[N_prediction] turn_rate_state_prediction;

  // Posterior predictive observations (including observation noise)
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;

  // Pointwise observation log likelihood for model comparison
  vector[2 * N_observed] log_likelihood;

  // Initialize forecasting from the final inferred latent state
  real x_previous = x_state[N_observed];
  real y_previous = y_state[N_observed];
  real speed_previous = speed_state[N_observed];
  real heading_previous = heading_state[N_observed];
  real turn_rate_previous = turn_rate_final;
  real time_previous = time_observed[N_observed];


  // ------------------------------------------------------------------
  // 1. POINTWISE OBSERVATION LOG LIKELIHOOD
  // ------------------------------------------------------------------

  for (n in 1:N_observed) {
    log_likelihood[n] = normal_lpdf(x_observed[n] | x_state[n], sigma_position_gps);
    log_likelihood[N_observed + n] = normal_lpdf(y_observed[n] | y_state[n], sigma_position_gps);
  }


  // ------------------------------------------------------------------
  // 2. POSTERIOR PREDICTIVE TRAJECTORY
  // ------------------------------------------------------------------

  // Sequentially propagate each posterior state draw into the future.
  for (n in 1:N_prediction) {
    real dt = time_prediction[n] - time_previous;

    // Conditional mean under the deterministic CTRV model
    vector[2] position = ctrv_position(
        dt,
        x_previous,
        y_previous,
        speed_previous,
        heading_previous,
        turn_rate_previous);

    // Sample future latent position from the process model
    real x_current = normal_rng(position[1], sigma_position_process * sqrt(dt));
    real y_current = normal_rng(position[2], sigma_position_process * sqrt(dt));

    // Sample future latent speed and preserve only the physical lower bound
    real speed_current = fmax(0, normal_rng(speed_previous, sigma_speed_process * sqrt(dt)));

    // Deterministic heading propagation
    real heading_current = heading_previous + turn_rate_previous * dt;

    // Keep the observation-derived forecast turn rate constant.
    real turn_rate_current = turn_rate_previous;

    // Store posterior predictive latent states
    x_state_prediction[n] = x_current;
    y_state_prediction[n] = y_current;
    speed_state_prediction[n] = speed_current;
    heading_state_prediction[n] = heading_current;
    turn_rate_state_prediction[n] = turn_rate_current;

    // Generate posterior predictive position observations
    x_observation_prediction[n] = normal_rng(x_current, sigma_position_gps);
    y_observation_prediction[n] = normal_rng(y_current, sigma_position_gps);

    // Continue forecasting from the current predictive state
    x_previous = x_current;
    y_previous = y_current;
    speed_previous = speed_current;
    heading_previous = heading_current;
    turn_rate_previous = turn_rate_current;
    time_previous = time_prediction[n];
  }
}

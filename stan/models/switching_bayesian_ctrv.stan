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

  // Mode-specific expected x, y, speed, and turn rate for one transition
  vector mode_transition_mean(
      int mode,
      real dt,
      real x_previous,
      real y_previous,
      real speed_previous,
      real heading_previous,
      real turn_rate_previous,
      real stop_speed_decay_time,
      real stop_turn_decay_time) {
    vector[4] expected_state;

    if (mode == 1) {
      // Stop: nearly constant position with decaying motion states
      expected_state[1] = x_previous;
      expected_state[2] = y_previous;
      expected_state[3] = speed_previous
                            * exp(-dt / stop_speed_decay_time);
      expected_state[4] = turn_rate_previous
                            * exp(-dt / stop_turn_decay_time);
    } else {
      // Cruise and maneuver share the CTRV mean and differ in process scale
      vector[2] expected_position = ctrv_position(
          dt,
          x_previous,
          y_previous,
          speed_previous,
          heading_previous,
          turn_rate_previous);
      expected_state[1] = expected_position[1];
      expected_state[2] = expected_position[2];
      expected_state[3] = speed_previous;
      expected_state[4] = turn_rate_previous;
    }

    return expected_state;
  }

  // Proper normal log density on a finite interval
  real bounded_normal_log_density(
      real value,
      real mean,
      real scale,
      real lower_bound,
      real upper_bound) {
    real log_normalizer = log_diff_exp(
        normal_lcdf(upper_bound | mean, scale),
        normal_lcdf(lower_bound | mean, scale));
    return normal_lpdf(value | mean, scale) - log_normalizer;
  }

  // Inverse-CDF draw from a normal distribution truncated to a finite interval
  real bounded_normal_rng(
      real mean,
      real scale,
      real lower_bound,
      real upper_bound) {
    real probability_lower = normal_cdf(lower_bound | mean, scale);
    real probability_upper = normal_cdf(upper_bound | mean, scale);

    // The fallback protects generated quantities from extreme CDF rounding.
    if (probability_upper <= probability_lower) {
      return fmin(upper_bound, fmax(lower_bound, mean));
    }

    {
      real probability = uniform_rng(probability_lower, probability_upper);
      real interior_probability = fmin(
          1 - 1e-12,
          fmax(1e-12, probability));
      real draw = mean + scale * inv_Phi(interior_probability);
      return fmin(upper_bound, fmax(lower_bound, draw));
    }
  }

  // Continuous-state transition log density conditional on one motion mode
  real mode_transition_log_density(
      int mode,
      real dt,
      real x_current,
      real y_current,
      real speed_current,
      real turn_rate_current,
      real x_previous,
      real y_previous,
      real speed_previous,
      real heading_previous,
      real turn_rate_previous,
      real sigma_position_process,
      real sigma_speed_process,
      real sigma_turn_rate_process,
      vector position_process_multiplier,
      vector speed_process_multiplier,
      vector turn_rate_process_multiplier,
      real stop_speed_decay_time,
      real stop_turn_decay_time,
      real turn_rate_limit) {
    vector[4] expected_state = mode_transition_mean(
        mode,
        dt,
        x_previous,
        y_previous,
        speed_previous,
        heading_previous,
        turn_rate_previous,
        stop_speed_decay_time,
        stop_turn_decay_time);
    real sqrt_dt = sqrt(dt);
    real position_scale = sigma_position_process
                          * position_process_multiplier[mode] * sqrt_dt;
    real speed_scale = sigma_speed_process
                       * speed_process_multiplier[mode] * sqrt_dt;
    real turn_rate_scale = sigma_turn_rate_process
                           * turn_rate_process_multiplier[mode] * sqrt_dt;
    real log_density = normal_lpdf(
        x_current | expected_state[1], position_scale);

    log_density += normal_lpdf(
        y_current | expected_state[2], position_scale);
    // Normalization matters because the competing modes use different scales.
    log_density += bounded_normal_log_density(
        speed_current,
        expected_state[3],
        speed_scale,
        0,
        100);
    log_density += bounded_normal_log_density(
        turn_rate_current,
        expected_state[4],
        turn_rate_scale,
        -turn_rate_limit,
        turn_rate_limit);

    return log_density;
  }
}

data {
  // Fixed semantic mapping: 1 = stop, 2 = cruise, 3 = maneuver
  int<lower=1> K;

  // Noisy position observations used as external-trajectory proxies
  int<lower=2> N_observed;
  vector[N_observed] time_observed;                 // elapsed time [s]
  vector[N_observed] x_observed;                    // local x position [m]
  vector[N_observed] y_observed;                    // local y position [m]

  // Prediction horizon
  int<lower=1> N_prediction;
  vector[N_prediction] time_prediction;             // future elapsed time [s]

  // Initial-state priors
  real x_initial_prior_mean;
  real y_initial_prior_mean;
  real<lower=0> position_initial_prior_scale;        // [m]
  real<lower=0> speed_initial_prior_mean;            // [m/s]
  real<lower=0> speed_initial_prior_scale;           // [m/s]
  real heading_initial_prior_mean;                   // [rad]
  real<lower=0> heading_initial_prior_scale;         // [rad]
  real turn_rate_initial_prior_mean;                 // [rad/s]
  real<lower=0> turn_rate_state_prior_scale;         // [rad/s]
  real<lower=0> turn_rate_limit;                     // absolute limit [rad/s]

  // Observation- and global process-noise prior scales
  real<lower=0> sigma_position_gps_prior_scale;      // [m]
  real<lower=0> sigma_position_process_prior_scale;  // [m/sqrt(s)]
  real<lower=0> sigma_speed_process_prior_scale;     // [(m/s)/sqrt(s)]
  real<lower=0> sigma_turn_rate_process_prior_scale; // [(rad/s)/sqrt(s)]

  // Markov-chain hyperparameters
  simplex[K] initial_mode_probability;
  array[K] vector<lower=0>[K] alpha_transition;

  // Fixed multipliers identify stop, cruise, and maneuver process scales
  vector<lower=0>[K] position_process_multiplier;
  vector<lower=0>[K] speed_process_multiplier;
  vector<lower=0>[K] turn_rate_process_multiplier;

  // Fixed stop-mode e-folding times [s]
  real<lower=0> stop_speed_decay_time;
  real<lower=0> stop_turn_decay_time;
}

transformed data {
  if (K != 3) {
    reject("K must equal 3: stop, cruise, and maneuver");
  }
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
  for (i in 1:K) {
    for (j in 1:K) {
      if (alpha_transition[i][j] <= 0) {
        reject("Every alpha_transition value must be positive");
      }
    }
  }
  if (!(position_process_multiplier[1] > 0
        && position_process_multiplier[1]
        < position_process_multiplier[2]
        && position_process_multiplier[2]
           < position_process_multiplier[3])) {
    reject("Position multipliers must increase from stop to maneuver");
  }
  if (!(speed_process_multiplier[1] > 0
        && speed_process_multiplier[1] < speed_process_multiplier[2]
        && speed_process_multiplier[2] < speed_process_multiplier[3])) {
    reject("Speed multipliers must increase from stop to maneuver");
  }
  if (!(turn_rate_process_multiplier[1] > 0
        && turn_rate_process_multiplier[1]
        < turn_rate_process_multiplier[2]
        && turn_rate_process_multiplier[2]
           < turn_rate_process_multiplier[3])) {
    reject("Turn-rate multipliers must increase from stop to maneuver");
  }
  if (stop_speed_decay_time <= 0 || stop_turn_decay_time <= 0) {
    reject("Stop decay times must be positive");
  }
}

parameters {
  // Continuous latent motion states
  vector[N_observed] x_state;
  vector[N_observed] y_state;
  // Zero is part of the model support; numerical initials must be interior.
  vector<lower=0, upper=100>[N_observed] speed_state;
  real heading_initial;
  vector<lower=-turn_rate_limit,
         upper=turn_rate_limit>[N_observed] turn_rate_state;

  // Each row is P(mode_t = j | mode_(t-1) = i)
  array[K] simplex[K] transition_probability;

  // Observation and global process noise
  real<lower=1e-6> sigma_position_gps;
  real<lower=1e-6> sigma_position_process;
  real<lower=1e-6> sigma_speed_process;
  real<lower=1e-6> sigma_turn_rate_process;
}

transformed parameters {
  vector[N_observed] heading_state;
  matrix[N_observed - 1, K] mode_transition_log_density_value;
  matrix[N_observed - 1, K] log_filtered_mode_probability;
  vector[N_observed - 1] forward_log_scale;

  // Unwrapped deterministic heading propagation
  heading_state[1] = heading_initial;
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];
    heading_state[n] = heading_state[n - 1]
                       + turn_rate_state[n - 1] * dt;
  }

  // Mode-specific transition emissions for m_t, t = 2, ..., N_observed
  for (r in 1:(N_observed - 1)) {
    int n = r + 1;
    real dt = time_observed[n] - time_observed[n - 1];
    for (k in 1:K) {
      mode_transition_log_density_value[r, k]
          = mode_transition_log_density(
              k,
              dt,
              x_state[n],
              y_state[n],
              speed_state[n],
              turn_rate_state[n],
              x_state[n - 1],
              y_state[n - 1],
              speed_state[n - 1],
              heading_state[n - 1],
              turn_rate_state[n - 1],
              sigma_position_process,
              sigma_speed_process,
              sigma_turn_rate_process,
              position_process_multiplier,
              speed_process_multiplier,
              turn_rate_process_multiplier,
              stop_speed_decay_time,
              stop_turn_decay_time,
              turn_rate_limit);
    }
  }

  // Scaled forward algorithm in log space
  {
    vector[K] log_weight;
    for (k in 1:K) {
      log_weight[k] = log(initial_mode_probability[k])
                      + mode_transition_log_density_value[1, k];
    }
    forward_log_scale[1] = log_sum_exp(log_weight);
    for (k in 1:K) {
      log_filtered_mode_probability[1, k]
          = log_weight[k] - forward_log_scale[1];
    }
  }
  if (N_observed > 2) {
    for (r in 2:(N_observed - 1)) {
      vector[K] log_weight;
      for (j in 1:K) {
        vector[K] predecessor_log_weight;
        for (i in 1:K) {
          predecessor_log_weight[i]
              = log_filtered_mode_probability[r - 1, i]
                + log(transition_probability[i][j]);
        }
        log_weight[j] = mode_transition_log_density_value[r, j]
                        + log_sum_exp(predecessor_log_weight);
      }
      forward_log_scale[r] = log_sum_exp(log_weight);
      for (k in 1:K) {
        log_filtered_mode_probability[r, k]
            = log_weight[k] - forward_log_scale[r];
      }
    }
  }
}

model {
  // Weakly informative priors for the first continuous state only
  x_state[1] ~ normal(x_initial_prior_mean, position_initial_prior_scale);
  y_state[1] ~ normal(y_initial_prior_mean, position_initial_prior_scale);
  speed_state[1] ~ normal(speed_initial_prior_mean,
                          speed_initial_prior_scale);
  heading_initial ~ normal(heading_initial_prior_mean,
                           heading_initial_prior_scale);
  turn_rate_state[1] ~ normal(turn_rate_initial_prior_mean,
                              turn_rate_state_prior_scale);

  // Persistent row-wise Dirichlet priors are supplied as data.
  for (i in 1:K) {
    transition_probability[i] ~ dirichlet(alpha_transition[i]);
  }

  // Half-normal priors due to positive sigma constraints
  sigma_position_gps ~ normal(0, sigma_position_gps_prior_scale);
  sigma_position_process ~ normal(0, sigma_position_process_prior_scale);
  sigma_speed_process ~ normal(0, sigma_speed_process_prior_scale);
  sigma_turn_rate_process ~ normal(0, sigma_turn_rate_process_prior_scale);

  // Sum over every discrete mode sequence via the forward recursion.
  target += sum(forward_log_scale);

  // The position-only observation model is shared by all modes.
  x_observed ~ normal(x_state, sigma_position_gps);
  y_observed ~ normal(y_state, sigma_position_gps);
}

generated quantities {
  // Smoothed posterior mode probabilities conditional on each continuous draw
  matrix[N_observed - 1, K] mode_probability;
  array[N_observed - 1] int<lower=1, upper=K> most_likely_mode;
  vector[K] final_filtered_mode_probability;

  // Future latent states and sampled future modes
  vector[N_prediction] x_state_prediction;
  vector[N_prediction] y_state_prediction;
  vector[N_prediction] speed_state_prediction;
  vector[N_prediction] heading_state_prediction;
  vector[N_prediction] turn_rate_state_prediction;
  array[N_prediction] int<lower=1, upper=K> mode_prediction;

  // Posterior predictive noisy position observations
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;

  // Pointwise position-observation log likelihood
  vector[2 * N_observed] log_likelihood;

  // Scaled backward recursion followed by smoothing
  {
    int mode_count = N_observed - 1;
    matrix[N_observed - 1, K] log_backward;

    for (k in 1:K) {
      log_backward[mode_count, k] = 0;
    }
    if (mode_count > 1) {
      for (reverse_index in 1:(mode_count - 1)) {
        int r = mode_count - reverse_index;
        for (i in 1:K) {
          vector[K] successor_log_weight;
          for (j in 1:K) {
            successor_log_weight[j]
                = log(transition_probability[i][j])
                  + mode_transition_log_density_value[r + 1, j]
                  + log_backward[r + 1, j];
          }
          log_backward[r, i] = log_sum_exp(successor_log_weight)
                               - forward_log_scale[r + 1];
        }
      }
    }

    for (r in 1:mode_count) {
      vector[K] log_smoothed_probability;
      for (k in 1:K) {
        log_smoothed_probability[k]
            = log_filtered_mode_probability[r, k] + log_backward[r, k];
      }
      mode_probability[r] = softmax(log_smoothed_probability)';
      most_likely_mode[r] = 1;
      for (k in 2:K) {
        if (mode_probability[r, k]
            > mode_probability[r, most_likely_mode[r]]) {
          most_likely_mode[r] = k;
        }
      }
    }
  }

  final_filtered_mode_probability = softmax(
      log_filtered_mode_probability[N_observed - 1]');

  for (n in 1:N_observed) {
    log_likelihood[n] = normal_lpdf(
        x_observed[n] | x_state[n], sigma_position_gps);
    log_likelihood[N_observed + n] = normal_lpdf(
        y_observed[n] | y_state[n], sigma_position_gps);
  }

  // Recursive future simulation starts at the final latent state.
  {
    real x_previous = x_state[N_observed];
    real y_previous = y_state[N_observed];
    real speed_previous = speed_state[N_observed];
    real heading_previous = heading_state[N_observed];
    real turn_rate_previous = turn_rate_state[N_observed];
    real time_previous = time_observed[N_observed];
    int previous_mode = categorical_rng(final_filtered_mode_probability);

    for (n in 1:N_prediction) {
      real dt = time_prediction[n] - time_previous;
      int current_mode = categorical_rng(
          transition_probability[previous_mode]);
      vector[4] expected_state = mode_transition_mean(
          current_mode,
          dt,
          x_previous,
          y_previous,
          speed_previous,
          heading_previous,
          turn_rate_previous,
          stop_speed_decay_time,
          stop_turn_decay_time);
      real sqrt_dt = sqrt(dt);
      real position_scale = sigma_position_process
                            * position_process_multiplier[current_mode]
                            * sqrt_dt;
      real speed_scale = sigma_speed_process
                         * speed_process_multiplier[current_mode] * sqrt_dt;
      real turn_rate_scale = sigma_turn_rate_process
                             * turn_rate_process_multiplier[current_mode]
                             * sqrt_dt;
      real x_current = normal_rng(expected_state[1], position_scale);
      real y_current = normal_rng(expected_state[2], position_scale);
      real speed_current = bounded_normal_rng(
          expected_state[3], speed_scale, 0, 100);
      real heading_current = heading_previous + turn_rate_previous * dt;
      real turn_rate_current = bounded_normal_rng(
          expected_state[4],
          turn_rate_scale,
          -turn_rate_limit,
          turn_rate_limit);

      x_state_prediction[n] = x_current;
      y_state_prediction[n] = y_current;
      speed_state_prediction[n] = speed_current;
      heading_state_prediction[n] = heading_current;
      turn_rate_state_prediction[n] = turn_rate_current;
      mode_prediction[n] = current_mode;
      x_observation_prediction[n] = normal_rng(
          x_current, sigma_position_gps);
      y_observation_prediction[n] = normal_rng(
          y_current, sigma_position_gps);

      x_previous = x_current;
      y_previous = y_current;
      speed_previous = speed_current;
      heading_previous = heading_current;
      turn_rate_previous = turn_rate_current;
      time_previous = time_prediction[n];
      previous_mode = current_mode;
    }
  }
}

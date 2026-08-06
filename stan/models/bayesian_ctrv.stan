functions {
  /**
   * Deterministic CTRV position-transition function.
   *
   * Computes the conditional mean of the next two-dimensional position
   * under the constant-turn-rate-and-velocity (CTRV) assumption.
   * Speed and turn rate are assumed constant over the interval dt.
   *
   * @param dt        Time interval [s].
   * @param x         Current latent x-position [m].
   * @param y         Current latent y-position [m].
   * @param speed     Current latent speed [m/s].
   * @param heading   Current latent heading [rad].
   * @param turn_rate Current latent turn rate [rad/s].
   *
   * @return Expected position after dt, before adding process noise [m].
   */
  vector ctrv_position(
      real dt,
      real x,
      real y,
      real speed,
      real heading,
      real turn_rate) {
    vector[2] position;

    if (abs(turn_rate) > 1e-6) {
      /*
       * Closed-form CTRV transition for non-zero turn rate.
       *
       * Under constant speed and turn rate, the vessel follows a
       * circular arc with instantaneous radius speed / turn_rate.
       */
      position[1] = x + speed / turn_rate * (sin(heading + turn_rate * dt) - sin(heading));
      position[2] = y + speed / turn_rate * (-cos(heading + turn_rate * dt) + cos(heading));
    } else {
      /*
       * Zero-turn-rate limit of the CTRV model.
       *
       * For a sufficiently small turn rate, straight-line motion is used.
       * The separate branch avoids numerical instability caused by division
       * by a value close to zero.
       */
      position[1] = x + speed * dt * cos(heading);
      position[2] = y + speed * dt * sin(heading);
    }

    return position;
  }
}


data {
  /*
   * Observed data D.
   *
   * Only two-dimensional positions are included in the likelihood.
   * The underlying motion states are treated as latent variables.
   */
  int<lower=2> N_observed;             // Number of observed time points.
  vector[N_observed] time_observed;    // Elapsed observation times [s].
  vector[N_observed] x_observed;       // Observed local x-positions [m].
  vector[N_observed] y_observed;       // Observed local y-positions [m].

  /*
   * Future time points used for posterior predictive forecasting.
   *
   * These values do not contribute to posterior inference because no
   * observations are available at these times.
   */
  int<lower=1> N_prediction;            // Number of future prediction steps.
  vector[N_prediction] time_prediction; // Future elapsed times [s].

  /*
   * Hyperparameters of the initial-state priors.
   *
   * These values specify prior knowledge about the initial latent state
   * before conditioning on the complete observed trajectory.
   */
  real x_initial_prior_mean;           // Prior mean of initial x-position [m].
  real y_initial_prior_mean;           // Prior mean of initial y-position [m].

  real<lower=0> position_initial_prior_scale;
  // Prior standard deviation shared by the initial x- and y-position [m].

  real<lower=0> speed_initial_prior_mean;
  // Prior mean of the initial latent speed [m/s].

  real<lower=0> speed_initial_prior_scale;
  // Prior standard deviation of the initial latent speed [m/s].

  real heading_initial_prior_mean;
  // Prior mean of the initial heading angle [rad].

  real<lower=0> heading_initial_prior_scale;
  // Prior standard deviation of the initial heading angle [rad].

  real turn_rate_initial_prior_mean;
  // Prior mean of the latent turn-rate states [rad/s].

  real<lower=0> turn_rate_state_prior_scale;
  // Prior standard deviation of the latent turn-rate states [rad/s].

  real<lower=0> turn_rate_limit;
  // Maximum admissible absolute turn rate [rad/s].

  /*
   * Hyperparameters of the prior distributions for the unknown
   * observation- and process-noise scales.
   *
   * Because the corresponding sigma parameters are constrained to be
   * positive, zero-centered normal priors induce half-normal priors.
   */
  real<lower=0> sigma_position_gps_prior_scale;
  // Scale of the half-normal prior for position-observation noise [m].

  real<lower=0> sigma_position_process_prior_scale;
  // Scale of the half-normal prior for position-process noise [m / sqrt(s)].

  real<lower=0> sigma_speed_process_prior_scale;
  // Scale of the half-normal prior for speed-process noise
  // [(m/s) / sqrt(s)].

  real<lower=0> sigma_turn_rate_process_prior_scale;
  // Scale of the half-normal prior for turn-rate-process noise
  // [(rad/s) / sqrt(s)].
}


transformed data {
  /*
   * Validate the temporal ordering required by the sequential
   * state-transition model.
   */
  for (n in 2:N_observed) {
    if (time_observed[n] <= time_observed[n - 1]) {
      reject("time_observed must be strictly increasing");
    }
  }

  /*
   * Posterior predictive forecasting must begin after the final
   * observed time point.
   */
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
  /*
   * Unknown latent motion states.
   *
   * These quantities are not directly observed. Their joint posterior
   * distribution is inferred from the position observations, the priors,
   * and the stochastic CTRV state-transition model.
   */
  vector[N_observed] x_state;          // Latent true x-position [m].
  vector[N_observed] y_state;          // Latent true y-position [m].

  /*
   * The lower bound permits stationary motion.
   * The upper bound restricts inference to physically plausible speeds.
   * Variational-inference initial values must lie strictly within the bounds.
   */
  vector<lower=0, upper=100>[N_observed] speed_state;
  // Latent speed at each observed time point [m/s].

  real heading_initial;
  // Unknown initial heading angle [rad].

  vector<lower=-turn_rate_limit,
         upper=turn_rate_limit>[N_observed] turn_rate_state;
  // Latent turn rate at each observed time point [rad/s].

  /*
   * Unknown global observation-noise parameter.
   *
   * This parameter describes the discrepancy between the observed
   * positions and the corresponding latent true positions.
   */
  real<lower=1e-6> sigma_position_gps;
  // Standard deviation of the position-observation model [m].

  /*
   * Unknown global process-noise parameters.
   *
   * These parameters quantify departures from the idealized CTRV dynamics.
   * They are expressed per square-root second so that transition variance
   * grows proportionally with the elapsed time interval.
   */
  real<lower=1e-6> sigma_position_process;
  // Position-process diffusion scale [m / sqrt(s)].

  real<lower=1e-6> sigma_speed_process;
  // Speed-process diffusion scale [(m/s) / sqrt(s)].

  real<lower=1e-6> sigma_turn_rate_process;
  // Turn-rate-process diffusion scale [(rad/s) / sqrt(s)].
}


transformed parameters {
  /*
   * Deterministically derived heading trajectory.
   *
   * Only the initial heading and the turn-rate trajectory are inferred
   * directly. Subsequent headings are obtained by integrating the
   * piecewise-constant latent turn rate.
   */
  vector[N_observed] heading_state;

  heading_state[1] = heading_initial;

  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];

    heading_state[n] =
        heading_state[n - 1]
        + turn_rate_state[n - 1] * dt;
  }
}


model {
  /*
   * BAYESIAN MODEL
   *
   * Stan constructs the unnormalized joint posterior density
   *
   *   p(latent states, noise parameters | observed positions)
   *
   * proportional to
   *
   *   initial-state priors
   *   * parameter priors
   *   * stochastic state-transition distributions
   *   * position-observation likelihood.
   *
   * The posterior distribution is therefore not written as one explicit
   * statement. It is induced by all probability statements in this block.
   */

  // ------------------------------------------------------------------
  // 1. PRIORS FOR THE INITIAL LATENT STATE
  // ------------------------------------------------------------------

  /*
   * Prior distribution of the initial latent position:
   *
   *   p(x_1, y_1)
   */
  x_state[1] ~ normal(
      x_initial_prior_mean,
      position_initial_prior_scale);

  y_state[1] ~ normal(
      y_initial_prior_mean,
      position_initial_prior_scale);

  /*
   * Prior distribution of the initial latent speed:
   *
   *   p(v_1)
   *
   * The parameter constraint truncates the normal distribution to
   * the physically admissible interval [0, 100].
   */
  speed_state[1] ~ normal(
      speed_initial_prior_mean,
      speed_initial_prior_scale);

  /*
   * Prior distribution of the initial heading:
   *
   *   p(heading_1)
   */
  heading_initial ~ normal(
      heading_initial_prior_mean,
      heading_initial_prior_scale);


  // ------------------------------------------------------------------
  // 2. PRIOR REGULARIZATION OF THE LATENT TURN-RATE TRAJECTORY
  // ------------------------------------------------------------------

  /*
   * Marginal prior for the complete latent turn-rate path:
   *
   *   p(turn_rate_1, ..., turn_rate_N)
   *
   * This prior regularizes every turn-rate state toward the specified
   * prior mean. The temporal dependence between consecutive turn rates
   * is additionally imposed by the state-transition model below.
   */
  turn_rate_state ~ normal(
      turn_rate_initial_prior_mean,
      turn_rate_state_prior_scale);


  // ------------------------------------------------------------------
  // 3. PRIORS FOR OBSERVATION- AND PROCESS-NOISE PARAMETERS
  // ------------------------------------------------------------------

  /*
   * Half-normal priors for positive standard-deviation parameters.
   *
   * The distributions are written as zero-centered normal priors.
   * Combined with the positive parameter constraints, they correspond
   * to half-normal prior distributions:
   *
   *   sigma ~ HalfNormal(prior_scale).
   */
  sigma_position_gps ~ normal(
      0,
      sigma_position_gps_prior_scale);

  sigma_position_process ~ normal(
      0,
      sigma_position_process_prior_scale);

  sigma_speed_process ~ normal(
      0,
      sigma_speed_process_prior_scale);

  sigma_turn_rate_process ~ normal(
      0,
      sigma_turn_rate_process_prior_scale);


  // ------------------------------------------------------------------
  // 4. STOCHASTIC STATE-TRANSITION MODEL / PROCESS MODEL
  // ------------------------------------------------------------------

  /*
   * The transition distributions define
   *
   *   p(state_n | state_{n-1}, process-noise parameters).
   *
   * They act as conditional priors for each subsequent latent state.
   * The deterministic CTRV equations provide the conditional mean,
   * while Gaussian process noise permits deviations from idealized
   * constant-turn-rate-and-velocity motion.
   */
  for (n in 2:N_observed) {
    real dt = time_observed[n] - time_observed[n - 1];

    /*
     * Conditional mean of the subsequent position under the
     * deterministic CTRV motion model.
     */
    vector[2] position = ctrv_position(
        dt,
        x_state[n - 1],
        y_state[n - 1],
        speed_state[n - 1],
        heading_state[n - 1],
        turn_rate_state[n - 1]);

    /*
     * Latent position-transition distribution.
     *
     * Position uncertainty increases with sqrt(dt), corresponding to
     * a diffusion process whose variance grows linearly with time.
     */
    x_state[n] ~ normal(
        position[1],
        sigma_position_process * sqrt(dt));

    y_state[n] ~ normal(
        position[2],
        sigma_position_process * sqrt(dt));

    /*
     * Latent speed-transition distribution.
     *
     * The speed follows a Gaussian random walk around the preceding
     * latent speed.
     */
    speed_state[n] ~ normal(
        speed_state[n - 1],
        sigma_speed_process * sqrt(dt));

    /*
     * Latent turn-rate-transition distribution.
     *
     * The turn rate follows a Gaussian random walk around the preceding
     * latent turn rate.
     */
    turn_rate_state[n] ~ normal(
        turn_rate_state[n - 1],
        sigma_turn_rate_process * sqrt(dt));
  }


  // ------------------------------------------------------------------
  // 5. LIKELIHOOD / POSITION-OBSERVATION MODEL
  // ------------------------------------------------------------------

  /*
   * Position-only likelihood:
   *
   *   p(x_observed, y_observed | latent positions, sigma_position_gps).
   *
   * The observed positions are modeled as noisy measurements of the
   * corresponding latent true positions. Conditional on the latent states
   * and observation-noise parameter, x- and y-errors are assumed Gaussian,
   * independent, and homoscedastic.
   *
   * Together with the priors and transition distributions above, this
   * likelihood updates prior uncertainty to the posterior distribution.
   */
  x_observed ~ normal(
      x_state,
      sigma_position_gps);

  y_observed ~ normal(
      y_state,
      sigma_position_gps);
}


generated quantities {
  /*
   * POSTERIOR-DERIVED QUANTITIES
   *
   * This block does not influence posterior inference.
   * It generates additional quantities conditional on posterior draws
   * obtained from the model block.
   */

  /*
   * Posterior predictive draws of future latent motion states.
   *
   * These variables represent possible future true trajectories and
   * therefore include process uncertainty but no observation noise.
   */
  vector[N_prediction] x_state_prediction;
  vector[N_prediction] y_state_prediction;
  vector[N_prediction] speed_state_prediction;
  vector[N_prediction] heading_state_prediction;
  vector[N_prediction] turn_rate_state_prediction;

  /*
   * Posterior predictive draws of future noisy position observations.
   *
   * These variables include both:
   *   1. uncertainty in the latent future trajectory, and
   *   2. observation uncertainty.
   *
   * They represent measurements that could be produced by the assumed
   * observation model at future time points.
   */
  vector[N_prediction] x_observation_prediction;
  vector[N_prediction] y_observation_prediction;

  /*
   * Pointwise observation log-likelihood contributions.
   *
   * These values contain only the position-observation likelihood,
   * not the prior or state-transition densities. They can be used for
   * predictive model-comparison methods such as LOO or WAIC.
   */
  vector[2 * N_observed] log_likelihood;

  /*
   * Initialize posterior predictive forecasting with the final
   * inferred latent state from the observation interval.
   *
   * Each posterior draw therefore produces its own future trajectory.
   */
  real x_previous = x_state[N_observed];
  real y_previous = y_state[N_observed];
  real speed_previous = speed_state[N_observed];
  real heading_previous = heading_state[N_observed];
  real turn_rate_previous = turn_rate_state[N_observed];
  real time_previous = time_observed[N_observed];


  // ------------------------------------------------------------------
  // POINTWISE OBSERVATION LOG LIKELIHOOD
  // ------------------------------------------------------------------

  for (n in 1:N_observed) {
    log_likelihood[n] =
        normal_lpdf(
            x_observed[n]
            | x_state[n],
              sigma_position_gps);

    log_likelihood[N_observed + n] =
        normal_lpdf(
            y_observed[n]
            | y_state[n],
              sigma_position_gps);
  }


  // ------------------------------------------------------------------
  // POSTERIOR PREDICTIVE TRAJECTORY
  // ------------------------------------------------------------------

  /*
   * Sequentially propagate the final latent posterior state into the
   * future using the same stochastic process model as during inference.
   */
  for (n in 1:N_prediction) {
    real dt = time_prediction[n] - time_previous;

    /*
     * Conditional mean of the future position under the deterministic
     * CTRV transition.
     */
    vector[2] position = ctrv_position(
        dt,
        x_previous,
        y_previous,
        speed_previous,
        heading_previous,
        turn_rate_previous);

    /*
     * Draw the subsequent latent position from the process model.
     */
    real x_current = normal_rng(
        position[1],
        sigma_position_process * sqrt(dt));

    real y_current = normal_rng(
        position[2],
        sigma_position_process * sqrt(dt));

    /*
     * Draw the subsequent latent speed and restrict it to the admissible
     * physical range used in the parameter block.
     */
    real speed_current = fmin(
        100,
        fmax(
            0,
            normal_rng(
                speed_previous,
                sigma_speed_process * sqrt(dt))));

    /*
     * Propagate the heading deterministically using the preceding
     * latent turn rate.
     */
    real heading_current =
        heading_previous
        + turn_rate_previous * dt;

    /*
     * Draw the subsequent latent turn rate and restrict it to the
     * admissible range.
     */
    real turn_rate_current = fmin(
        turn_rate_limit,
        fmax(
            -turn_rate_limit,
            normal_rng(
                turn_rate_previous,
                sigma_turn_rate_process * sqrt(dt))));

    /*
     * Store the posterior predictive latent-state draw.
     */
    x_state_prediction[n] = x_current;
    y_state_prediction[n] = y_current;
    speed_state_prediction[n] = speed_current;
    heading_state_prediction[n] = heading_current;
    turn_rate_state_prediction[n] = turn_rate_current;

    /*
     * Generate posterior predictive position observations by adding
     * observation noise to the predicted latent positions.
     */
    x_observation_prediction[n] = normal_rng(
        x_current,
        sigma_position_gps);

    y_observation_prediction[n] = normal_rng(
        y_current,
        sigma_position_gps);

    /*
     * Use the current predictive state as the starting state for the
     * next forecasting step.
     */
    x_previous = x_current;
    y_previous = y_current;
    speed_previous = speed_current;
    heading_previous = heading_current;
    turn_rate_previous = turn_rate_current;
    time_previous = time_prediction[n];
  }
}

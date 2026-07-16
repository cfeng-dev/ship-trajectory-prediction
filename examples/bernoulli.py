"""Estimate a Bernoulli success probability with CmdStanPy."""

from cmdstanpy import CmdStanModel

from ship_trajectory_prediction.paths import project_path

# Path to the Stan model
STAN_FILE = project_path("stan/examples/bernoulli.stan")


def main():
    # Example binary observations:
    # 1 = success
    # 0 = failure
    observations = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]

    # Count observed successes and failures
    num_observations = len(observations)
    num_successes = sum(observations)
    num_failures = num_observations - num_successes

    # Prior parameters:
    # Beta(1, 1) is a uniform prior over theta
    alpha_prior = 1
    beta_prior = 1

    prior_mean_success = alpha_prior / (alpha_prior + beta_prior)
    prior_mean_failure = 1 - prior_mean_success

    print("=" * 60)
    print("Bayesian Bernoulli Model")
    print("=" * 60)

    print(f"Number of observations : {num_observations}")
    print(f"Observed successes     : {num_successes}")
    print(f"Observed failures      : {num_failures}")

    print("\nPrior information:")
    print(f"Prior distribution     : Beta({alpha_prior}, {beta_prior})")
    print(f"Prior mean success     : {prior_mean_success:.3f}")
    print(f"Prior mean failure     : {prior_mean_failure:.3f}")

    print(f"\nStan model             : {STAN_FILE}")

    # Compile the Stan model
    model = CmdStanModel(stan_file=str(STAN_FILE))

    # Data passed to Stan
    data = {
        "N": num_observations,
        "y": observations,
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
    }

    # Draw posterior samples
    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=500,
        iter_sampling=1000,
        seed=42,
    )

    # Extract posterior samples of theta
    # theta represents the probability of success, i.e. P(y = 1)
    theta_samples = fit.stan_variable("theta")

    posterior_mean_success = theta_samples.mean()
    posterior_mean_failure = 1 - posterior_mean_success

    print("\nPosterior summary:")
    print(fit.summary())

    print("\nPosterior mean probabilities:")
    print(f"Success probability P(y = 1): {posterior_mean_success:.3f}")
    print(f"Failure probability P(y = 0): {posterior_mean_failure:.3f}")


if __name__ == "__main__":
    main()

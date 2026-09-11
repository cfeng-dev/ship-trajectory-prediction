"""Terminal reporting for configured Bayesian CTRV priors."""

from statistics import NormalDist

import numpy as np

import bayestraj.models.bayesian_ctrv as bayesian_model


def print_prior_report(priors: bayesian_model.BayesianCTRVPriors) -> None:
    """Print configured prior assumptions and their derivations."""
    speed_z = NormalDist().inv_cdf(1.0 - priors.speed_prior_tail_probability / 2.0)
    turn_z = NormalDist().inv_cdf(1.0 - priors.turn_rate_prior_tail_probability / 2.0)
    turn_threshold = priors.turn_rate_prior_abs_rate_deg_s
    turn_scale = float(np.rad2deg(priors.turn_rate_prior_scale))
    print("Bayessche CTRV-Priorverteilungen\n================================")
    _normal_report("Anfangsgeschwindigkeit v_1", "v_1", priors.speed_prior_upper_mps, "m/s", priors.speed_prior_tail_probability, priors.speed_prior_scale, speed_z, "Halbnormal", scale_symbol="v")
    print("Anfangskurswinkel theta_1\n  Verteilung: theta_1 ~ Gleichverteilung(-pi, pi)\n  Konfiguriert:\n    theta_1 in [-pi, pi] rad = [-180, 180] deg\n  Berechnung:\n    p(theta_1) = 1 / (pi - (-pi)) = 0.159155 1/rad\n    p(theta_1) = 1 / (180 - (-180)) = 0.00277778 1/deg\n  Ergebnis:\n    Alle Anfangsrichtungen sind gleich wahrscheinlich.\n")
    print(f"Initiale Drehrate omega_1\n  Verteilung: omega_1 ~ Normal(0, s_omega)\n  Konfiguriert:\n    P(|omega_1| > {turn_threshold:g} deg/s) = {priors.turn_rate_prior_tail_probability:g}\n  Berechnung:\n    omega_max = {turn_threshold:g} deg/s\n    z = Phi^-1(1 - {priors.turn_rate_prior_tail_probability:g} / 2) = {turn_z:.6g}\n    s_omega = {turn_threshold:g} / Phi^-1(1 - {priors.turn_rate_prior_tail_probability:g} / 2) = {turn_scale:.6g} deg/s\n    s_omega = {turn_scale:.6g} * pi / 180 = {priors.turn_rate_prior_scale:.6g} rad/s\n  Ergebnis:\n    omega_1 ~ Normal(0, {priors.turn_rate_prior_scale:.6g} rad/s)\n")
    _exponential_report("Positionsmessrauschen", "sigma_obs", priors.sigma_position_observation_prior_upper_m, "m", priors.sigma_position_observation_prior_tail_probability, priors.sigma_position_observation_prior_rate, "1/m")
    _exponential_report("Geschwindigkeits-Prozessrauschen", "sigma_v", priors.sigma_speed_process_prior_upper_mps, "m/s", priors.sigma_speed_process_prior_tail_probability, priors.sigma_speed_process_prior_rate, "s/m")
    upper = priors.sigma_turn_rate_process_prior_upper_deg_s
    upper_rad = float(np.deg2rad(upper))
    rate = priors.sigma_turn_rate_process_prior_rate
    print(f"Drehraten-Prozessrauschen sigma_omega\n  Verteilung: sigma_omega ~ Exponential(lambda)\n  Konfiguriert:\n    P(sigma_omega > {upper:g} deg/s) = {priors.sigma_turn_rate_process_prior_tail_probability:g}\n  Berechnung:\n    P(X > x) = exp(-lambda * x)\n    {upper:g} deg/s * pi / 180 = {upper_rad:.6g} rad/s\n    exp(-lambda * {upper_rad:.6g}) = {priors.sigma_turn_rate_process_prior_tail_probability:g}\n    lambda = -ln({priors.sigma_turn_rate_process_prior_tail_probability:g}) / {upper_rad:.6g} = {rate:.6g} s/rad\n    E[sigma_omega] = 1 / lambda = {1.0 / rate:.6g} rad/s ({np.rad2deg(1.0 / rate):.6g} deg/s)\n  Ergebnis:\n    sigma_omega ~ Exponential(lambda = {rate:.6g} s/rad)")


def _normal_report(label, symbol, upper, unit, tail, scale, quantile, distribution, *, scale_unit=None, scale_symbol=None):
    scale_unit = scale_unit or unit
    scale_symbol = scale_symbol or symbol
    print(f"{label}\n  Verteilung: {symbol} ~ {distribution}(s_{scale_symbol})\n  Konfiguriert:\n    P({symbol} > {upper:g} {unit}) = {tail:g}\n  Berechnung:\n    z = Phi^-1(1 - {tail:g} / 2) = {quantile:.6g}\n    s_{scale_symbol} = {upper:g} / Phi^-1(1 - {tail:g} / 2) = {scale:.6g} {scale_unit}\n  Ergebnis:\n    {symbol} ~ {distribution}(s_{scale_symbol} = {scale:.6g} {scale_unit})\n")


def _exponential_report(label, symbol, upper, unit, tail, rate, rate_unit):
    print(f"{label} {symbol}\n  Verteilung: {symbol} ~ Exponential(lambda)\n  Konfiguriert:\n    P({symbol} > {upper:g} {unit}) = {tail:g}\n  Berechnung:\n    P(X > x) = exp(-lambda * x)\n    exp(-lambda * {upper:g}) = {tail:g}\n    lambda = -ln({tail:g}) / {upper:g} = {rate:.6g} {rate_unit}\n    E[{symbol}] = 1 / lambda = {1.0 / rate:.6g} {unit}\n  Ergebnis:\n    {symbol} ~ Exponential(lambda = {rate:.6g} {rate_unit})\n")

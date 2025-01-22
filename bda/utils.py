import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


# Beta-Binomial
def plot_beta_prior(alpha: float, beta: float):
    """
    Plot the Beta distribution.

    Parameters:
    alpha (float): Alpha parameter of the Beta distribution.
    beta (float): Beta parameter of the Beta distribution.
    """

    x = np.linspace(0, 1, 1000)
    y = stats.beta.pdf(x, alpha, beta)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"Beta(alpha={alpha}, beta={beta})")
    plt.title("Beta Prior")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_binomial_likelihood(n: int, k: int):
    """
    Plot the likelihood for a Binomial distribution given n and k.

    Parameters:
    n (int): Number of trials.
    k (int): Number of successes.
    """

    p = np.linspace(0, 1, 1000)
    likelihood = stats.binom.pmf(k, n, p)
    likelihood /= likelihood.sum()  # Scale for comparison so area under the curve is 1
    likelihood *= len(p) - 1

    plt.figure(figsize=(8, 5))
    plt.plot(p, likelihood, label=f"Scaled Likelihood (n={n}, k={k})")
    plt.title("Binomial Likelihood")
    plt.xlabel("Probability")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_beta_binomial(alpha: float, beta: float, k: int, n: int):
    """
    Plot the prior, scaled likelihood, and posterior distributions for a Beta-Binomial model.

    Parameters:
    alpha (float): Prior alpha parameter of the Beta distribution.
    beta (float): Prior beta parameter of the Beta distribution.
    k (int): Number of successes.
    n (int): Number of trials.
    """

    x = np.linspace(0, 1, 10000)

    # Prior distribution
    prior_pdf = stats.beta.pdf(x, alpha, beta)

    # Scaled likelihood (Binomial likelihood scaled to match the Beta shape)
    likelihood_pdf = x**k * (1 - x) ** (n - k)
    likelihood_pdf /= likelihood_pdf.sum()  # Scale for comparison
    likelihood_pdf *= len(x) - 1

    # Posterior distribution
    posterior_alpha = alpha + k
    posterior_beta = beta + (n - k)
    posterior_pdf = stats.beta.pdf(x, posterior_alpha, posterior_beta)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior_pdf, label="Prior", linestyle="--")
    plt.plot(x, likelihood_pdf, label="Scaled Likelihood", linestyle=":")
    plt.plot(x, posterior_pdf, label="Posterior", linestyle="-")

    plt.title("Beta-Binomial Analysis")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_beta(alpha: float, beta: float):
    """
    Summarize the Beta distribution given its parameters.

    Parameters:
    alpha (float): Alpha parameter of the Beta distribution.
    beta (float): Beta parameter of the Beta distribution.

    Returns:
    pd.DataFrame: Summary statistics for the Beta distribution.
    """
    # Calculate summary statistics for the Beta distribution
    mean = alpha / (alpha + beta)
    mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    sd = np.sqrt(var)

    # Create a DataFrame to summarize results
    summary = pd.DataFrame(
        {
            "alpha": [alpha],
            "beta": [beta],
            "mean": [mean],
            "mode": [mode],
            "var": [var],
            "sd": [sd],
        },
        index=["Beta Distribution"],
    )

    summary.index.name = "distribution"

    return summary


def summarize_beta_binomial(alpha: float, beta: float, k: int, n: int):
    """
    Summarize the prior and posterior Beta distributions for a Beta-Binomial model.

    Parameters:
    alpha (float): Prior alpha parameter of the Beta distribution.
    beta (float): Prior beta parameter of the Beta distribution.
    k (int): Number of successes.
    n (int): Number of trials.

    Returns:
    pd.DataFrame: Summary statistics for the prior and posterior distributions.
    """
    # Prior distribution parameters
    prior_mean = alpha / (alpha + beta)
    prior_mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan
    prior_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    prior_sd = np.sqrt(prior_var)

    # Posterior distribution parameters
    posterior_alpha = alpha + k
    posterior_beta = beta + (n - k)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    posterior_mode = (
        (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
        if posterior_alpha > 1 and posterior_beta > 1
        else np.nan
    )
    posterior_var = (posterior_alpha * posterior_beta) / (
        (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
    )
    posterior_sd = np.sqrt(posterior_var)

    # Create a DataFrame to summarize results
    summary = pd.DataFrame(
        {
            "alpha": [alpha, posterior_alpha],
            "beta": [beta, posterior_beta],
            "mean": [prior_mean, posterior_mean],
            "mode": [prior_mode, posterior_mode],
            "var": [prior_var, posterior_var],
            "sd": [prior_sd, posterior_sd],
        },
        index=["prior", "posterior"],
    )

    summary.index.name = "model"

    return summary

    return summary


# Gamma-Poisson
def plot_gamma_prior(shape: float, rate: float):
    """
    Plot the Gamma distribution.

    Parameters:
    shape (float): Shape parameter of the Gamma distribution.
    rate (float): Rate parameter of the Gamma distribution.
    """

    x = np.linspace(0, 5 * shape / rate, 1000)
    y = stats.gamma.pdf(x, shape, scale=1 / rate)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"Gamma(shape={shape}, rate={rate})")
    plt.title("Gamma Prior")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_poisson_likelihood(counts: np.ndarray, lambda_upper_bound: float):
    """
    Plot the joint likelihood of the Poisson distribution for given counts and lambda values.

    Parameters:
    counts (np.ndarray): Array of observed counts.
    lambda_upper_bound (float): Upper bound for lambda values to evaluate.
    """

    lambdas = np.linspace(0, lambda_upper_bound, 1000)
    likelihoods = []
    for lambda_ in lambdas:
        joint_likelihood = np.prod(stats.poisson.pmf(counts, lambda_))
        likelihoods.append(joint_likelihood)

    likelihoods = np.array(likelihoods)
    likelihoods /= likelihoods.sum()  # Scale for comparison
    likelihoods *= len(lambdas) - 1

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, likelihoods, label="Joint Likelihood")
    plt.title("Poisson Joint Likelihood")
    plt.xlabel("Lambda")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gamma_poisson(shape: float, rate: float, sum_y: int, n: int):
    """
    Plot the prior, scaled likelihood, and posterior distributions for a Gamma-Poisson model.

    Parameters:
    shape (float): Shape parameter of the Gamma distribution.
    rate (float): Rate parameter of the Gamma distribution.
    sum_y (int): Sum of observed counts.
    n (int): Number of observations.
    """

    x = np.linspace(0, 5 * shape / rate, 1000)

    # Prior distribution
    prior_pdf = stats.gamma.pdf(x, shape, scale=1 / rate)

    # Scaled likelihood (Poisson likelihood scaled to match Gamma shape)
    likelihood_pdf = x**sum_y * np.exp(-n * x)
    likelihood_pdf /= likelihood_pdf.sum()  # Scale for comparison
    likelihood_pdf *= len(x) - 1

    # Posterior distribution
    posterior_shape = shape + sum_y
    posterior_rate = rate + n
    posterior_pdf = stats.gamma.pdf(x, posterior_shape, scale=1 / posterior_rate)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior_pdf, label="Prior", linestyle="--")
    plt.plot(x, likelihood_pdf, label="Scaled Likelihood", linestyle=":")
    plt.plot(x, posterior_pdf, label="Posterior", linestyle="-")

    plt.title("Gamma-Poisson Analysis")
    plt.xlabel("Rate")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_gamma_poisson(shape: float, rate: float, sum_y: int, n: int):
    """
    Summarize the prior and posterior Gamma distributions for a Gamma-Poisson model.

    Parameters:
    shape (float): Shape parameter of the Gamma distribution.
    rate (float): Rate parameter of the Gamma distribution.
    sum_y (int): Sum of observed counts.
    n (int): Number of observations.

    Returns:
    pd.DataFrame: Summary statistics for the prior and posterior distributions.
    """

    # Prior distribution parameters
    prior_mean = shape / rate
    prior_var = shape / rate**2
    prior_sd = np.sqrt(prior_var)

    # Posterior distribution parameters
    posterior_shape = shape + sum_y
    posterior_rate = rate + n
    posterior_mean = posterior_shape / posterior_rate
    posterior_var = posterior_shape / posterior_rate**2
    posterior_sd = np.sqrt(posterior_var)

    # Create a DataFrame to summarize results
    summary = pd.DataFrame(
        {
            "shape": [shape, posterior_shape],
            "rate": [rate, posterior_rate],
            "mean": [prior_mean, posterior_mean],
            "var": [prior_var, posterior_var],
            "sd": [prior_sd, posterior_sd],
        },
        index=["prior", "posterior"],
    )

    summary.index.name = "model"

    return summary

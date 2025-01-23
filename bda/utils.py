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
    plt.xlabel("π")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_binomial_likelihood(n: int, y: int):
    """
    Plot the likelihood for a Binomial distribution given n and k.

    Parameters:
    n (int): Number of trials.
    y (int): Number of successes.
    """

    p = np.linspace(0, 1, 1000)
    likelihood = stats.binom.pmf(y, n, p)
    likelihood /= np.trapezoid(likelihood, p)  # Scale for comparison

    plt.figure(figsize=(8, 5))
    plt.plot(p, likelihood, label=f"Scaled Likelihood (n={n}, y={y})")
    plt.title("Binomial Likelihood")
    plt.xlabel("π")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_beta_binomial(alpha: float, beta: float, n: int, y: int):
    """
    Plot the prior, scaled likelihood, and posterior distributions for a Beta-Binomial model.

    Parameters:
    alpha (float): Prior alpha parameter of the Beta distribution.
    beta (float): Prior beta parameter of the Beta distribution.
    n (int): Number of trials.
    y (int): Number of successes.
    """

    x = np.linspace(0, 1, 10000)

    # Prior distribution
    prior_pdf = stats.beta.pdf(x, alpha, beta)

    # Scaled likelihood (Binomial likelihood scaled to match the Beta shape)
    likelihood_pdf = x**y * (1 - x) ** (n - y)
    likelihood_pdf /= np.trapezoid(likelihood_pdf, x)  # Scale for comparison

    # Posterior distribution
    posterior_alpha = alpha + y
    posterior_beta = beta + (n - y)
    posterior_pdf = stats.beta.pdf(x, posterior_alpha, posterior_beta)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior_pdf, label="Prior", linestyle="--")
    plt.plot(x, likelihood_pdf, label="Scaled Likelihood", linestyle=":")
    plt.plot(x, posterior_pdf, label="Posterior", linestyle="-")

    plt.title("Beta-Binomial Analysis")
    plt.xlabel("π")
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


def summarize_beta_binomial(alpha: float, beta: float, n: int, y: int):
    """
    Summarize the prior and posterior Beta distributions for a Beta-Binomial model.

    Parameters:
    alpha (float): Prior alpha parameter of the Beta distribution.
    beta (float): Prior beta parameter of the Beta distribution.
    n (int): Number of trials.
    y (int): Number of successes.

    Returns:
    pd.DataFrame: Summary statistics for the prior and posterior distributions.
    """
    # Prior distribution parameters
    prior_mean = alpha / (alpha + beta)
    prior_mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan
    prior_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    prior_sd = np.sqrt(prior_var)

    # Posterior distribution parameters
    posterior_alpha = alpha + y
    posterior_beta = beta + (n - y)
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


# Gamma-Poisson
def plot_gamma_prior(shape: float, rate: float):
    """
    Plot the Gamma distribution.

    Parameters:
    shape (float): Shape parameter of the Gamma distribution.
    rate (float): Rate parameter of the Gamma distribution.
    """

    x_max = stats.gamma.ppf(0.99999, shape, scale=1 / rate)
    x = np.linspace(0, x_max, 1000)
    y = stats.gamma.pdf(x, shape, scale=1 / rate)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"Gamma(shape={shape}, rate={rate})")
    plt.title("Gamma Prior")
    plt.xlabel("λ")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_poisson_likelihood(y: np.ndarray, lambda_upper_bound: float):
    """
    Plot the joint likelihood of the Poisson distribution for given data (counts) and lambda values.

    Parameters:
    y (np.ndarray): Array of observed data (counts).
    lambda_upper_bound (float): Upper bound for lambda values to evaluate.
    """

    lambdas = np.linspace(0, lambda_upper_bound, 1000)
    likelihoods = []
    for lambda_ in lambdas:
        joint_likelihood = np.prod(stats.poisson.pmf(y, lambda_))
        likelihoods.append(joint_likelihood)

    likelihoods = np.array(likelihoods)
    likelihoods /= np.trapezoid(likelihoods, lambdas)  # Scale for comparison

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, likelihoods, label="Scaled Joint Likelihood")
    plt.title("Poisson Joint Likelihood")
    plt.xlabel("λ")
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

    # calculate range of x
    x_min = min(
        (
            stats.gamma.ppf(1e-05, shape, scale=1 / rate),
            stats.gamma.ppf(1e-05, shape + sum_y, scale=1 / (rate + n)),
            stats.gamma.ppf(1e-05, sum_y + 1, scale=1 / n),
        )
    )
    x_max = max(
        (
            stats.gamma.ppf(0.99999, shape, scale=1 / rate),
            stats.gamma.ppf(0.99999, shape + sum_y, scale=1 / (rate + n)),
            stats.gamma.ppf(0.99999, sum_y + 1, scale=1 / n),
        )
    )
    x = np.linspace(x_min, x_max, 1000)

    # Prior distribution
    prior_pdf = stats.gamma.pdf(x, shape, scale=1 / rate)

    # Scaled likelihood (Poisson likelihood scaled to match Gamma shape)
    likelihood_pdf = x**sum_y * np.exp(-n * x)
    likelihood_pdf /= np.trapezoid(likelihood_pdf, x)  # Scale for comparison

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
    plt.xlabel("λ")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_gamma(shape: float, rate: float) -> pd.DataFrame:
    """
    Summarize the Gamma distribution given its parameters.

    Parameters
    ----------
    shape : float
        Shape parameter of the Gamma distribution.
    rate : float
        Rate parameter of the Gamma distribution.

    Returns
    -------
    pd.DataFrame
        Summary statistics for the Gamma distribution.
    """
    # Mean of Gamma(shape, rate) = shape / rate
    mean = shape / rate

    # Mode of Gamma(shape, rate) = (shape - 1) / rate, if shape > 1; otherwise undefined/NaN
    mode = (shape - 1) / rate if shape > 1 else np.nan

    # Variance of Gamma(shape, rate) = shape / (rate^2)
    var = shape / (rate**2)

    # Standard deviation
    sd = np.sqrt(var)

    # Create a DataFrame to summarize results
    summary = pd.DataFrame(
        {
            "shape": [shape],
            "rate": [rate],
            "mean": [mean],
            "mode": [mode],
            "var": [var],
            "sd": [sd],
        },
        index=["Gamma Distribution"],
    )

    summary.index.name = "distribution"
    return summary


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

    # --- Prior parameters & derived stats ---
    prior_mean = shape / rate
    prior_var = shape / rate**2
    prior_sd = np.sqrt(prior_var)

    # Mode of the prior
    if shape > 1:
        prior_mode = (shape - 1) / rate
    else:
        # If shape <= 1, the peak is at 0
        prior_mode = 0.0

    # --- Posterior parameters & derived stats ---
    posterior_shape = shape + sum_y
    posterior_rate = rate + n
    posterior_mean = posterior_shape / posterior_rate
    posterior_var = posterior_shape / (posterior_rate**2)
    posterior_sd = np.sqrt(posterior_var)

    # Mode of the posterior
    if posterior_shape > 1:
        posterior_mode = (posterior_shape - 1) / posterior_rate
    else:
        posterior_mode = 0.0

    # --- Build summary DataFrame ---
    summary = pd.DataFrame(
        {
            "shape": [shape, posterior_shape],
            "rate": [rate, posterior_rate],
            "mean": [prior_mean, posterior_mean],
            "mode": [prior_mode, posterior_mode],
            "var": [prior_var, posterior_var],
            "sd": [prior_sd, posterior_sd],
        },
        index=["prior", "posterior"],
    )
    summary.index.name = "model"

    return summary


# Normal-Normal
def plot_normal_prior(mean: float, sd: float):
    """
    Plot the Normal (Gaussian) distribution as a prior.

    Parameters
    ----------
    mean : float
        Mean of the Normal distribution.
    sd : float
        Standard deviation of the Normal distribution.
    """

    # Calculate the x-range based on quantiles (1e-05 and 0.99999)
    x_min = stats.norm.ppf(1e-05, loc=mean, scale=sd)
    x_max = stats.norm.ppf(0.99999, loc=mean, scale=sd)

    # Generate the x values
    x = np.linspace(x_min, x_max, 1000)

    # Calculate the Normal PDF
    y = stats.norm.pdf(x, loc=mean, scale=sd)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"Normal(mean={mean}, sd={sd})")
    plt.title("Normal Prior")
    plt.xlabel("μ")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_normal_likelihood(y: np.ndarray, sigma: float):
    """
    Plot the joint likelihood of the Normal distribution for given data (y)
    and a known standard deviation (sigma). The parameter of interest here is
    the mean (mu).

    Parameters
    ----------
    y : np.ndarray
        Array of observed data.
    sigma : float
        Known standard deviation of the Normal distribution.
    """

    # Calculate summary statistics
    y_mean = np.mean(y)
    y_sd = np.std(y, ddof=1)  # sample standard deviation
    n = len(y)

    # Calculate search range for mu
    x_min = y_mean - 4 * y_sd / np.sqrt(n)
    x_max = y_mean + 4 * y_sd / np.sqrt(n)

    # Create grid for mu values
    mus = np.linspace(x_min, x_max, 1000)

    # Calculate the (unscaled) joint likelihood for each mu in the grid
    likelihoods = []
    for mu in mus:
        # Joint likelihood under Normal with mean=mu and std=sigma
        joint_likelihood = np.prod(stats.norm.pdf(y, loc=mu, scale=sigma))
        likelihoods.append(joint_likelihood)

    # Convert to numpy array
    likelihoods = np.array(likelihoods)

    # Scale the likelihood for plotting
    # Here, we use trapezoidal numerical integration to scale the curve
    area = np.trapezoid(likelihoods, mus)
    if area > 0:
        likelihoods /= area

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(mus, likelihoods, label="Scaled Joint Likelihood")
    plt.title("Normal Joint Likelihood")
    plt.xlabel("μ")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_normal_normal(
    prior_mean: float, prior_sd: float, y_sigma: float, y_mean: float, n: int
):
    """
    Plot the prior, scaled likelihood, and posterior for a Normal-Normal model.

    Parameters
    ----------
    prior_mean : float
        Mean of the prior distribution (Normal).
    prior_sd : float
        Standard deviation of the prior distribution (Normal).
    y_sigma : float
        Known standard deviation of each observation in the data.
    y_mean : float
        Sample mean of the observed data.
    n : int
        Number of observations in the sample.
    """

    # ----------------------------
    # 1) Compute posterior parameters
    # ----------------------------
    # Posterior variance and mean for mu | (y_mean)
    post_var = 1.0 / (1.0 / (prior_sd**2) + n / (y_sigma**2))
    post_sd = np.sqrt(post_var)
    post_mean = post_var * (prior_mean / (prior_sd**2) + (n * y_mean) / (y_sigma**2))

    # ----------------------------
    # 2) Likelihood distribution
    # ----------------------------
    # For the sample mean y_mean, the likelihood for mu is:
    # L(mu) ∝ Normal(y_mean | mu, (y_sigma^2 / n))
    # The standard deviation of y_mean is y_sigma / sqrt(n).
    like_sd = y_sigma / np.sqrt(n)

    # ----------------------------
    # 3) Determine plotting range
    #    We'll capture the central mass from:
    #      - the prior
    #      - the posterior
    #      - the "likelihood" distribution (centered at y_mean)
    #    using the normal ppf at 1e-5 and 0.99999
    # ----------------------------
    x_min = min(
        stats.norm.ppf(1e-5, loc=prior_mean, scale=prior_sd),
        stats.norm.ppf(1e-5, loc=post_mean, scale=post_sd),
        stats.norm.ppf(1e-5, loc=y_mean, scale=like_sd),
    )
    x_max = max(
        stats.norm.ppf(0.99999, loc=prior_mean, scale=prior_sd),
        stats.norm.ppf(0.99999, loc=post_mean, scale=post_sd),
        stats.norm.ppf(0.99999, loc=y_mean, scale=like_sd),
    )

    x = np.linspace(x_min, x_max, 1000)

    # ----------------------------
    # 4) Compute PDFs
    # ----------------------------
    # Prior PDF
    prior_pdf = stats.norm.pdf(x, loc=prior_mean, scale=prior_sd)

    # Likelihood PDF (in terms of mu, centered at y_mean)
    # We'll scale it so the area is comparable to a PDF for the plot
    like_pdf_raw = stats.norm.pdf(y_mean, loc=x, scale=like_sd)
    # Use trapezoid to normalize
    like_area = np.trapezoid(like_pdf_raw, x)
    if like_area > 0:
        like_pdf = like_pdf_raw / like_area
    else:
        like_pdf = like_pdf_raw  # fallback if area is zero or extremely small

    # Posterior PDF
    post_pdf = stats.norm.pdf(x, loc=post_mean, scale=post_sd)

    # ----------------------------
    # 5) Plot: Prior, scaled likelihood, posterior
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior_pdf, "r--", label="Prior")
    plt.plot(x, like_pdf, "g:", label="Scaled Likelihood")
    plt.plot(x, post_pdf, "b-", label="Posterior")

    plt.title("Normal-Normal Analysis")
    plt.xlabel("μ")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_normal(mean: float, sd: float) -> pd.DataFrame:
    """
    Summarize a Normal distribution given its mean and standard deviation.

    Parameters
    ----------
    mean : float
        Mean (μ) of the Normal distribution.
    sd : float
        Standard deviation (τ) of the Normal distribution.

    Returns
    -------
    pd.DataFrame
        Summary statistics for the Normal distribution, including
        mean, mode, variance, and standard deviation.
    """
    var = sd**2
    mode = mean  # For a Normal(μ, σ²), the mode = mean

    summary = pd.DataFrame(
        {
            "mean": [mean],
            "mode": [mode],
            "var": [var],
            "sd": [sd],
        },
        index=["Normal Distribution"],
    )
    summary.index.name = "distribution"

    return summary


def summarize_normal_normal(
    prior_mean: float, prior_sd: float, y_sigma: float, y_mean: float, n: int
) -> pd.DataFrame:
    """
    Summarize the prior and posterior Normal distributions for a Normal–Normal model.

    Parameters
    ----------
    prior_mean : float
        Mean (μ) of the prior Normal distribution.
    prior_sd : float
        Standard deviation (τ) of the prior Normal distribution.
    y_sigma : float
        Known standard deviation (σ) of the observations.
    y_mean : float
        Sample mean of the observed data (ȳ).
    n : int
        Number of observations.

    Returns
    -------
    pd.DataFrame
        Summary statistics for the prior and posterior distributions,
        including mean, mode, variance, and standard deviation.
    """
    # ----------------------------
    # Prior
    # ----------------------------
    prior_var = prior_sd**2
    prior_mode = prior_mean  # mode = mean for a Normal distribution

    # ----------------------------
    # Posterior (Normal-Normal Conjugacy)
    # ----------------------------
    # Posterior variance = [ 1/σ₀² + n/σ² ]⁻¹
    post_var = 1.0 / (1.0 / prior_var + n / (y_sigma**2))
    post_sd = np.sqrt(post_var)
    # Posterior mean = post_var * [ (μ₀ / σ₀²) + (n * ȳ / σ²) ]
    post_mean = post_var * ((prior_mean / prior_var) + (n * y_mean) / (y_sigma**2))
    post_mode = post_mean  # For a Normal, the mode = mean

    # ----------------------------
    # Build the DataFrame
    # ----------------------------
    summary = pd.DataFrame(
        {
            "mean": [prior_mean, post_mean],
            "mode": [prior_mode, post_mode],
            "var": [prior_var, post_var],
            "sd": [prior_sd, post_sd],
        },
        index=["prior", "posterior"],
    )
    summary.index.name = "model"

    return summary



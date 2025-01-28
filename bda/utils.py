import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import stan
import nest_asyncio
import arviz as az
from typing import Optional, Dict
import bambi as bmb

nest_asyncio.apply()


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
    likelihood /= np.trapz(likelihood, p)  # Scale for comparison

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
    likelihood_pdf /= np.trapz(likelihood_pdf, x)  # Scale for comparison

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
    likelihoods /= np.trapz(likelihoods, lambdas)  # Scale for comparison

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
    likelihood_pdf /= np.trapz(likelihood_pdf, x)  # Scale for comparison

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
    area = np.trapz(likelihoods, mus)
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
    like_area = np.trapz(like_pdf_raw, x)
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


# MCMC
def mcmc_stan(
    model_code,
    data,
    num_chains=4,
    num_samples=10000,
    num_warmup=5000,
    save_warmup: bool = False,
    random_seed=42,
    return_all: bool = False,
):
    """
    Run Markov Chain Monte Carlo (MCMC) sampling using Stan and process results.

    This function builds a Stan model, performs MCMC sampling, and converts
    the sampling results into a pandas DataFrame for easy analysis.

    Parameters:
    -----------
    model_code : str
        Stan model code defining the probabilistic model
    data : dict
        Input data dictionary for the Stan model
    num_chains : int, optional
        Number of parallel Markov chains to run (default: 4)
    num_samples : int, optional
        Number of samples to draw per chain after warmup (default: 1000)
    num_warmup : int, optional
        Number of warmup/initialization samples to discard (default: 500)
    save_warmup : bool, optional
        Whether to include warmup samples in the output (default: False)
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility (default: 42)
    return_all : bool, optional
        If set to True -> additonally return fit object and model itself

    Returns:
    --------
    pd.DataFrame
        DataFrame containing MCMC sampling results with columns:
        - chain_{i}: Samples for each chain
        - sample_nr: Sample number within the chain
        - var_name: Parameter name
        - is_warmup: Boolean indicating if sample is from warmup period
    fit: only if return_all is True
    model: only if return_all is true
    """

    # Build Stan model with specified data and random seed
    posterior = stan.build(model_code, data=data, random_seed=random_seed)

    # Perform MCMC sampling
    fit = posterior.sample(
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        save_warmup=save_warmup,
    )

    # Process and convert sampling results to DataFrame
    dfs = []
    for i, chains in enumerate(fit._draws[-len(fit.constrained_param_names) :]):
        # Extract variable name for current parameter
        var = fit.constrained_param_names[i]

        # Create DataFrame for current parameter's chains
        df = pd.DataFrame(
            chains, columns=[f"chain_{i}" for i in range(chains.shape[1])]
        )
        df["sample_nr"] = np.arange(chains.shape[0]) - (
            num_warmup if save_warmup else 0
        )
        df["var_name"] = var
        df["is_warmup"] = False

        # Mark warmup samples if save_warmup is True
        if save_warmup:
            df["is_warmup"].values[:num_warmup] = True

        dfs.append(df)

    # Concatenate results from all parameters
    dfs = pd.concat(dfs, ignore_index=True)

    if return_all:
        # return all
        return dfs, fit, posterior

    return dfs


def mcmc_trace(vars_n_chains: pd.DataFrame, variables=None):
    """
    Create trace plots for MCMC sampling results in a grid.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from MCMC_STAN with columns:
          chain_{i}, sample_nr, var_name, is_warmup
    variables : str or list of str, optional
        If None (default), all unique variables will be plotted.
        If a string, only that single variable is plotted.
        If a list of strings, only those variables are plotted.
    """
    # 1. Decide which variables to plot
    if variables is None:
        plot_vars = vars_n_chains["var_name"].unique()
    elif isinstance(variables, str):
        plot_vars = [variables]
    else:
        # Assume it's an iterable of strings
        plot_vars = variables

    # 2. Calculate grid dimensions
    num_vars = len(plot_vars)
    num_cols = int(min(3, num_vars))
    num_rows = int(np.ceil(num_vars / num_cols))

    # 3. Create subplots
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(7 * num_cols, 4 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # 4. Plot each variable
    for i, var in enumerate(plot_vars):
        var_data = vars_n_chains[vars_n_chains["var_name"] == var]
        if var_data.empty:
            # If this variable is not found, skip or show empty plot
            axes_flat[i].set_title(f"{var} - no data")
            axes_flat[i].set_xlabel("Sample Number")
            axes_flat[i].set_ylabel("Value")
            continue

        # Select chain columns
        chain_cols = [col for col in var_data.columns if col.startswith("chain_")]

        # Melt the DataFrame for seaborn
        melted_data = var_data.melt(
            id_vars=["sample_nr", "is_warmup"],
            value_vars=chain_cols,
            var_name="chain",
            value_name="value",
        )

        # Create trace plot
        sns.lineplot(
            data=melted_data,
            x="sample_nr",
            y="value",
            hue="chain",
            ax=axes_flat[i],
            alpha=0.5,
        )

        axes_flat[i].set_title(var)
        axes_flat[i].set_xlabel("Sample Number")
        axes_flat[i].set_ylabel("Value")

    # 5. Remove extra subplots if any
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle("MCMC trace")
    plt.tight_layout()
    plt.show()


def mcmc_hist(
    vars_n_chains: pd.DataFrame, variables=None, stat: str = "count", bins: int = 50
):
    """
    Create histplots for MCMC sampling results in a grid.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from MCMC_STAN with columns: chain_{i}, sample_nr, var_name, is_warmup
    variables : str or list of str, optional
        If None (default), all unique variables will be plotted.
        If a string, only that single variable is plotted.
        If a list of strings, only those variables are plotted.
    stat : str
        Statistic on y-axis (e.g., 'count', 'density', 'probability', etc.)
    bins : int
        Number of bins in histogram
    """
    # 1. Decide which variables to plot
    if variables is None:
        plot_vars = vars_n_chains["var_name"].unique()
    elif isinstance(variables, str):
        plot_vars = [variables]
    else:
        plot_vars = variables

    # 2. Calculate grid dimensions
    num_vars = len(plot_vars)
    num_cols = int(min(3, num_vars))
    num_rows = int(np.ceil(num_vars / num_cols))

    # 3. Create subplots
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(7 * num_cols, 4 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # 4. Plot each variable
    for i, var in enumerate(plot_vars):
        var_data = vars_n_chains[vars_n_chains["var_name"] == var]
        if var_data.empty:
            axes_flat[i].set_title(f"{var} - no data")
            axes_flat[i].set_xlabel(var)
            axes_flat[i].set_ylabel(stat)
            continue

        # Select chain columns
        chain_cols = [col for col in var_data.columns if col.startswith("chain_")]

        # Melt the DataFrame for seaborn
        melted_data = var_data.melt(
            id_vars=["sample_nr", "is_warmup"],
            value_vars=chain_cols,
            var_name="chain",
            value_name="value",
        )

        # Create histogram
        sns.histplot(data=melted_data, x="value", ax=axes_flat[i], stat=stat, bins=bins)
        axes_flat[i].set_title(f"dist of var {var}")
        axes_flat[i].set_xlabel(var)
        axes_flat[i].set_ylabel(stat)

    # 5. Remove extra subplots if any
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle("MCMC distribution (Histogram)")
    plt.tight_layout()
    plt.show()


def mcmc_combine_chains(vars_n_chains: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format MCMC data (chain_0, chain_1, etc.) into a long format
    with a single 'chain' column and a single 'value' column.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        A DataFrame from MCMC sampling with columns:
        - chain_{i} for each chain
        - sample_nr
        - var_name
        - is_warmup (optional)
        - and possibly other metadata columns

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns such as:
        - var_name
        - sample_nr
        - chain
        - value
        - is_warmup
        Any additional columns (non-chain) from the original DataFrame are preserved
        as "id_vars".
    """
    # Identify all columns that start with "chain_"
    chain_cols = [col for col in vars_n_chains.columns if col.startswith("chain_")]

    # Use 'melt' to pivot from wide to long format
    melted_df = vars_n_chains.melt(
        id_vars=[col for col in vars_n_chains.columns if col not in chain_cols],
        value_vars=chain_cols,
        var_name="chain",
        value_name="value",
    )

    return melted_df


def mcmc_dens(
    vars_n_chains: pd.DataFrame, variables=None, individual_chains: bool = True
):
    """
    Create density (kde) plots for MCMC sampling results in a grid.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from MCMC_STAN with columns: chain_{i}, sample_nr, var_name, is_warmup
    variables : str or list of str, optional
        If None (default), all unique variables will be plotted.
        If a string, only that single variable is plotted.
        If a list of strings, only those variables are plotted.
    individual_chains : bool
        If True, each chain is plotted with a different color (hue).
    """
    # 1. Decide which variables to plot
    if variables is None:
        plot_vars = vars_n_chains["var_name"].unique()
    elif isinstance(variables, str):
        plot_vars = [variables]
    else:
        plot_vars = variables

    # 2. Calculate grid dimensions
    num_vars = len(plot_vars)
    num_cols = int(min(3, num_vars))
    num_rows = int(np.ceil(num_vars / num_cols))

    # 3. Create subplots
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(7 * num_cols, 4 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # 4. Plot each variable
    for i, var in enumerate(plot_vars):
        var_data = vars_n_chains[vars_n_chains["var_name"] == var]
        if var_data.empty:
            axes_flat[i].set_title(f"{var} - no data")
            axes_flat[i].set_xlabel(var)
            axes_flat[i].set_ylabel("density")
            continue

        # Select chain columns
        chain_cols = [col for col in var_data.columns if col.startswith("chain_")]

        # Melt the DataFrame for seaborn
        melted_data = var_data.melt(
            id_vars=["sample_nr", "is_warmup"],
            value_vars=chain_cols,
            var_name="chain",
            value_name="value",
        )

        # Create density (kde) plot
        sns.kdeplot(
            data=melted_data,
            x="value",
            ax=axes_flat[i],
            hue="chain" if individual_chains else None,
            common_norm=False,
        )
        axes_flat[i].set_title(f"dist of var {var}")
        axes_flat[i].set_xlabel(var)
        axes_flat[i].set_ylabel("density")

    # 5. Remove extra subplots if any
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle("MCMC distribution (KDE)")
    plt.tight_layout()
    plt.show()


def mcmc_summarize(
    vars_n_chains: pd.DataFrame,
    filter_warmup: bool = True,
    percentiles: list = [0.025, 0.975],
) -> pd.DataFrame:
    """
    Summarize MCMC samples for each variable (combining all chains).

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame containing MCMC samples with columns:
        - chain_{i} for each chain
        - sample_nr
        - var_name
        - is_warmup (optional)
    filter_warmup : bool, optional
        If True, excludes warmup samples from the statistics (default: True)
    percentiles : list, optional
        A list of percentiles to compute (default: [0.025, 0.975])

    Returns
    -------
    pd.DataFrame
        A DataFrame with summary statistics for each variable:
        - n_samples
        - mean
        - mode (computed using KDE)
        - std
        - Configurable percentiles as separate columns
    """

    # 1. Optionally filter out warmup samples
    if filter_warmup and "is_warmup" in vars_n_chains.columns:
        df = vars_n_chains[~vars_n_chains["is_warmup"]]
    else:
        df = vars_n_chains

    # 2. Get unique variables
    unique_vars = df["var_name"].unique()

    # 3. Prepare summary statistics
    summary_list = []
    for var in unique_vars:
        # Select only rows for this variable
        var_data = df[df["var_name"] == var]

        # Identify chain columns
        chain_cols = [col for col in var_data.columns if col.startswith("chain_")]

        # Flatten all chain samples into a single 1D array
        samples = var_data[chain_cols].to_numpy().flatten()

        # Number of samples
        n_samples = len(samples)

        # Compute statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples, ddof=1)
        percentile_vals = np.percentile(samples, percentiles)

        # Compute mode using KDE
        kde = stats.gaussian_kde(samples)
        x_vals = np.linspace(min(samples), max(samples), 1000)
        kde_vals = kde(x_vals)
        mode_val = x_vals[np.argmax(kde_vals)]

        # Prepare row dictionary
        summary_row = {
            "var_name": var,
            "n_samples": n_samples,
            "mean": mean_val,
            "mode": mode_val,
            "std": std_val,
        }

        # Add percentile columns dynamically
        for p, val in zip(percentiles, percentile_vals):
            summary_row[f"p{float(p*100)}"] = val

        summary_list.append(summary_row)

    # 4. Convert to DataFrame
    summary_df = pd.DataFrame(summary_list).set_index("var_name")

    return summary_df


def neff_ratio(vars_n_chains: pd.DataFrame, filter_warmup: bool = True) -> pd.DataFrame:
    """
    Calculate the effective sample size (ESS) using ArviZ for MCMC sampling results.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from mcmc_stan with columns: chain_{i}, sample_nr, var_name, is_warmup
    filter_warmup : bool, optional
        If True, exclude warmup samples from the ESS calculation (default: True)

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by variable name, containing:
        - ess_mean (the ArviZ mean ESS)
        - ess_mean_ratio (the mean ESS as a fraction of total post-warmup draws)
    """

    # 1. Filter out warmup samples if requested
    samples = (
        vars_n_chains[~vars_n_chains["is_warmup"]] if filter_warmup else vars_n_chains
    )

    # Identify chain columns and variables
    chain_cols = [col for col in samples.columns if col.startswith("chain_")]
    unique_vars = samples["var_name"].unique()

    # 2. Build a dictionary where each key is a variable name and
    #    the corresponding value is a 2D array with shape (n_chains, n_draws)
    posterior_dict = {}
    for var in unique_vars:
        var_data = samples[samples["var_name"] == var].sort_values("sample_nr")
        # shape = (n_draws, n_chains)
        arr = var_data[chain_cols].to_numpy()
        # transpose to match ArviZ format: (n_chains, n_draws)
        posterior_dict[var] = arr.T

    # 3. Convert to InferenceData
    idata = az.from_dict(posterior=posterior_dict)

    # 4. Calculate the mean ESS with ArviZ
    ess_dataset = az.ess(idata, var_names=list(unique_vars), method="mean")
    # ess_dataset is now an xarray.Dataset that may be 0D if there's only one variable

    # 5. Extract each variable’s ESS value and build the final DataFrame
    rows = []
    for var in unique_vars:
        # If only one variable, .values is 0D; if multiple, .values is also dimensionless (1D with length=1).
        ess_val = ess_dataset[
            var
        ].values.item()  # Convert the array/scalar to a Python float
        n_chains, n_draws = posterior_dict[var].shape
        total_draws = n_chains * n_draws

        rows.append(
            {
                "var_name": var,
                "n_draws": n_draws,
                "n_chains": n_chains,
                "ess": ess_val,
                "ess_ratio": ess_val / total_draws if total_draws else 0,
            }
        )

    result_df = pd.DataFrame(rows).set_index("var_name")
    return result_df


def mcmc_acf(vars_n_chains: pd.DataFrame, lags: int = 20, filter_warmup: bool = True):
    """
    Calculate and plot autocorrelation for MCMC sampling results.

    Parameters:
    -----------
    vars_n_chains : pd.DataFrame
        DataFrame from mcmc_stan with columns: chain_{i}, sample_nr, var_name, is_warmup
    lags : int, optional
        Maximum number of lags to calculate (default: 20)
    filter_warmup : bool
        if true the warmup samples are getting filtered out
    """

    # Filter out warmup samples
    samples = (
        vars_n_chains[~vars_n_chains["is_warmup"]] if filter_warmup else vars_n_chains
    )

    # Get unique variables
    unique_vars = samples["var_name"].unique()

    # Calculate grid dimensions
    num_vars = len(unique_vars)
    num_cols = min(3, num_vars)
    num_rows = int(np.ceil(num_vars / num_cols))

    # Create subplots
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(7 * num_cols, 4 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Store autocorrelation results
    acf_results = {}

    # Plot for each variable
    for i, var in enumerate(unique_vars):
        # Filter data for current variable
        var_data = samples[samples["var_name"] == var]

        # Select chain columns
        chain_cols = [col for col in var_data.columns if col.startswith("chain_")]

        # Calculate autocorrelation for each chain
        chain_acf = []
        for chain_col in chain_cols:
            chain_samples = var_data[chain_col]

            # Calculate autocorrelation
            acf = np.correlate(
                chain_samples - chain_samples.mean(),
                chain_samples - chain_samples.mean(),
                mode="full",
            )[len(chain_samples) - 1 :]

            # Normalize
            acf /= acf[0]
            chain_acf.append(acf[: lags + 1])

        # Average across chains
        avg_acf = np.mean(chain_acf, axis=0)
        acf_results[var] = avg_acf

        # Plot
        axes_flat[i].stem(range(len(avg_acf)), avg_acf)
        axes_flat[i].set_title(f"ACF for {var}")
        axes_flat[i].set_xlabel("Lag")
        axes_flat[i].set_ylabel("Autocorrelation")
        axes_flat[i].axhline(y=0, color="r", linestyle="--")

    # Remove extra subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle("MCMC Autocorrelation")
    plt.tight_layout()
    plt.show()


def rhat(vars_n_chains: pd.DataFrame, filter_warmup: bool = True) -> pd.DataFrame:
    """
    Calculate R-hat (Gelman-Rubin statistic) for MCMC sampling results using ArviZ.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from mcmc_stan with columns: chain_{i}, sample_nr, var_name, is_warmup
    filter_warmup : bool, optional
        If True, exclude warmup samples from the R-hat calculation (default: True)

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by variable name, containing:
        - rhat: the Gelman-Rubin statistic for each parameter
        - n_chains: how many chains were used
    """

    # 1. Filter out warmup samples if requested
    samples = (
        vars_n_chains[~vars_n_chains["is_warmup"]] if filter_warmup else vars_n_chains
    )

    # Identify chain columns and unique variables
    chain_cols = [col for col in samples.columns if col.startswith("chain_")]
    unique_vars = samples["var_name"].unique()

    # 2. Build a dictionary of var_name -> 2D NumPy array (shape: [n_chains, n_draws])
    posterior_dict = {}
    for var in unique_vars:
        var_data = samples[samples["var_name"] == var].sort_values("sample_nr")
        # shape = (n_draws, n_chains)
        arr = var_data[chain_cols].to_numpy()
        # Transpose so that shape = (n_chains, n_draws), matching ArviZ's expected format
        posterior_dict[var] = arr.T

    # 3. Convert our dict into an InferenceData object
    idata = az.from_dict(posterior=posterior_dict)

    # 4. Calculate R-hat with ArviZ
    #    This returns an xarray.Dataset with a variable for each requested var_name
    rhat_dataset = az.rhat(idata, var_names=list(unique_vars))

    # 5. Build final results DataFrame
    rows = []
    for var in unique_vars:
        n_chains, _ = posterior_dict[var].shape

        # If there's only one variable or only one element, you'll get a 0D array;
        # either way, .item() extracts the scalar.
        rhat_val = rhat_dataset[var].values.item()

        rows.append(
            {
                "var_name": var,
                "rhat": rhat_val,
                "n_chains": n_chains,
            }
        )

    result_df = pd.DataFrame(rows).set_index("var_name")
    return result_df


# Metropolis-Hastings MCMC
def one_mh_iteration(
    proposal_model_gen: callable, prior_mul_likelihood: callable, current_loc: float
) -> pd.DataFrame:
    """
    Perform one iteration of the Metropolis-Hastings algorithm for sampling.

    This function implements the core Metropolis-Hastings sampling step, which
    decides whether to accept or reject a proposed location based on its likelihood
    and the proposal distribution.

    Parameters
    ----------
    proposal_model_gen : callable
        A function that generates a probability distribution (proposal model)
        centered at a given location.

    prior_mul_likelihood : callable
        A function that calculates the unnormalized likelihood (prior * likelihood)
        for a given location.

    current_loc : float
        The current location in the sampling process.

    Returns
    -------
    DataFrame
        The selected location after the Metropolis-Hastings iteration
        (either the proposed or current location) with alpha and proposed location.

    Notes
    -----
    The algorithm works by:
    1. Generating a proposal from a proposal distribution
    2. Calculating the acceptance probability based on likelihood ratios
    3. Randomly accepting or rejecting the proposal
    """

    # Generate proposal distribution centered at current location
    proposal_model = proposal_model_gen(current_loc)

    # Sample a new location from the proposal distribution
    proposed_loc = proposal_model.rvs(1)[0]

    # Generate proposal distribution for the proposed location
    proposal_model_proposed = proposal_model_gen(proposed_loc)

    # Calculate likelihoods for current and proposed locations
    current_loc_likelihood = prior_mul_likelihood(current_loc)
    proposed_loc_likelihood = prior_mul_likelihood(proposed_loc)

    # Calculate proposal probabilities for reverse transitions
    proba_current_loc = proposal_model_proposed.pdf(current_loc)
    proba_proposed_loc = proposal_model.pdf(proposed_loc)

    # Compute acceptance probability (Metropolis-Hastings ratio)
    acceptance_ratio = (proposed_loc_likelihood / current_loc_likelihood) * (
        proba_current_loc / proba_proposed_loc
    )
    alpha = min(1, acceptance_ratio)

    # Randomly accept or reject the proposed location
    next_loc = np.random.choice([proposed_loc, current_loc], p=[alpha, 1 - alpha])

    # return dataframe
    return pd.DataFrame(
        {"proposal": [proposed_loc], "alpha": [alpha], "next_loc": [next_loc]}
    )


def mh_tour(
    proposal_model_gen: callable,
    prior_mul_likelihood: callable,
    initial_loc: float,
    num_iterations: int,
) -> pd.DataFrame:
    """
    Perform a Metropolis-Hastings sampling tour.

    Parameters
    ----------
    proposal_model_gen : callable
        Function generating proposal distribution for a location.

    prior_mul_likelihood : callable
        Function calculating unnormalized likelihood for a location.

    initial_loc : float
        Starting location for the Markov chain.

    num_iterations : int
        Number of iterations to perform.

    Returns
    -------
    pd.DataFrame
        Concatenated trace of Metropolis-Hastings iterations.
    """
    # Initialize trace with initial location
    current_loc = initial_loc
    trace = []

    # Perform Metropolis-Hastings iterations
    for _ in range(num_iterations):
        # Perform one iteration and store result
        iteration_result = one_mh_iteration(
            proposal_model_gen, prior_mul_likelihood, current_loc
        )
        trace.append(iteration_result)

        # Update current location for next iteration
        current_loc = iteration_result["next_loc"].values[0]

    # Concatenate all iteration results
    return pd.concat(trace, ignore_index=True)


def MH_MCMC_uniform_proposal(
    prior_mul_likelihood: callable,
    half_width: float = 0.1,
    initial_loc: float = 1e-5,
    num_iterations: int = 1000,
    num_chains: int = 4,
    num_warmup: int = 500,
) -> list:
    """
    Perform Metropolis-Hastings MCMC sampling with uniform proposal distribution.

    Parameters
    ----------
    prior_mul_likelihood : callable
        Function calculating unnormalized likelihood for a location.

    half_width : float
        Half-width of uniform proposal distribution.

    initial_loc : float
        Starting location for each Markov chain.

    num_iterations : int
        Number of iterations per chain.

    num_chains : int
        Number of independent Markov chains to run.

    num_warmup : int
        Number of warmup steps (gets deleted)

    Returns
    -------
    list
        List of DataFrames, each representing a chain's trace.
    """

    def uniform_proposal_gen(loc):
        """Generate uniform proposal distribution centered at given location."""
        return stats.uniform(loc - half_width, 2 * half_width)

    # Run multiple chains
    traces = []
    for i in range(num_chains):
        # create a tour
        chain_trace = mh_tour(
            uniform_proposal_gen,
            prior_mul_likelihood,
            initial_loc,
            num_iterations + num_warmup,
        )[["next_loc"]]

        chain_trace.columns = [f"chain_{i}"]

        # append chain
        traces.append(chain_trace)

    # concat chains
    traces = pd.concat(traces, axis=1)

    # cut off warmup steps
    traces = traces.iloc[num_warmup:].reset_index(drop=True)

    traces["sample_nr"] = range(len(traces))
    traces["var_name"] = "pi"
    traces["is_warmup"] = False

    return traces


def MH_MCMC_uniform_independence_sampling(
    prior_mul_likelihood: callable,
    min_max: tuple = (0, 1),
    initial_loc: float = 0.5,
    num_iterations: int = 1000,
    num_chains: int = 4,
    num_warmup: int = 500,
) -> list:
    """
    Perform Metropolis-Hastings MCMC independence sampling alogorithm (proposal model doesn't depend on current location).

    Parameters
    ----------
    prior_mul_likelihood : callable
        Function calculating unnormalized likelihood for a location.

    min_max : tuple
        Min max of uniform model

    initial_loc : float
        Starting location for each Markov chain.

    num_iterations : int
        Number of iterations per chain.

    num_chains : int
        Number of independent Markov chains to run.

    num_warmup : int
        Number of warmup steps (gets deleted)

    Returns
    -------
    list
        List of DataFrames, each representing a chain's trace.
    """

    def uniform_proposal_gen(loc):
        """Generate uniform proposal distribution between min and max"""
        return stats.uniform(min_max[0], min_max[1])

    # Run multiple chains
    traces = []
    for i in range(num_chains):
        # create a tour
        chain_trace = mh_tour(
            uniform_proposal_gen,
            prior_mul_likelihood,
            initial_loc,
            num_iterations + num_warmup,
        )[["next_loc"]]

        chain_trace.columns = [f"chain_{i}"]

        # append chain
        traces.append(chain_trace)

    # concat chains
    traces = pd.concat(traces, axis=1)

    # cut off warmup steps
    traces = traces.iloc[num_warmup:].reset_index(drop=True)

    traces["sample_nr"] = range(len(traces))
    traces["var_name"] = "pi"
    traces["is_warmup"] = False

    return traces


# Posterior Inference & Prediction
## Posterior Estimation
def quantiles_beta(alpha: float, beta: float, quantiles: np.ndarray):
    """
    Calculate quantiles for a Beta distribution.

    Parameters:
    -----------
    alpha : float
        Alpha parameter of the Beta distribution
    beta : float
        Beta parameter of the Beta distribution
    quantiles : np.ndarray
        Array of probabilities for which to compute quantiles

    Returns:
    --------
    np.ndarray
        Quantile values corresponding to the input probabilities
    """

    return stats.beta.ppf(quantiles, alpha, beta)


def quantiles_gamma(shape: float, rate: float, quantiles: np.ndarray):
    """
    Calculate quantiles for a Gamma distribution.

    Parameters:
    -----------
    shape : float
        Shape parameter of the Gamma distribution
    rate : float
        Rate parameter of the Gamma distribution
    quantiles : np.ndarray
        Array of probabilities for which to compute quantiles

    Returns:
    --------
    np.ndarray
        Quantile values corresponding to the input probabilities
    """

    return stats.gamma.ppf(quantiles, shape, scale=1 / rate)


def quantiles_normal(mean: float, sd: float, quantiles: np.ndarray):
    """
    Calculate quantiles for a Normal distribution.

    Parameters:
    -----------
    mean : float
        Mean of the Normal distribution
    sd : float
        Standard deviation of the Normal distribution
    quantiles : np.ndarray
        Array of probabilities for which to compute quantiles

    Returns:
    --------
    np.ndarray
        Quantile values corresponding to the input probabilities
    """

    return stats.norm.ppf(quantiles, loc=mean, scale=sd)


def proba_beta(
    alpha: float,
    beta: float,
    pi_l_than: float = None,
    pi_between: np.ndarray = None,
    pi_h_than: float = None,
):
    """
    Calculate probabilities for different regions of a Beta distribution.

    Parameters:
    -----------
    alpha : float
        Alpha parameter of the Beta distribution
    beta : float
        Beta parameter of the Beta distribution
    pi_l_than : float, optional
        Upper bound for "less than" probability
    pi_between : np.ndarray, optional
        Array of [lower, upper] bounds for "between" probability
    pi_h_than : float, optional
        Lower bound for "higher than" probability

    Returns:
    --------
    tuple
        (P(X < pi_l_than), P(pi_between[0] < X < pi_between[1]), P(X > pi_h_than))
        Returns None for any probability where the corresponding input is None
    """

    # Validate pi_between if provided
    if pi_between is not None:
        if len(pi_between) != 2:
            raise ValueError("pi_between must be an array of exactly 2 values")

    # Calculate requested probabilities
    p_less = stats.beta.cdf(pi_l_than, alpha, beta) if pi_l_than is not None else None

    p_between = (
        (
            stats.beta.cdf(pi_between[1], alpha, beta)
            - stats.beta.cdf(pi_between[0], alpha, beta)
        )
        if pi_between is not None
        else None
    )

    p_higher = (
        (1 - stats.beta.cdf(pi_h_than, alpha, beta)) if pi_h_than is not None else None
    )

    return p_less, p_between, p_higher


def proba_gamma(
    shape: float,
    rate: float,
    pi_l_than: float = None,
    pi_between: np.ndarray = None,
    pi_h_than: float = None,
):
    """
    Calculate probabilities for different regions of a Gamma distribution.

    Parameters:
    -----------
    shape : float
        Shape parameter of the Gamma distribution
    rate : float
        Rate parameter of the Gamma distribution
    pi_l_than : float, optional
        Upper bound for "less than" probability
    pi_between : np.ndarray, optional
        Array of [lower, upper] bounds for "between" probability
    pi_h_than : float, optional
        Lower bound for "higher than" probability

    Returns:
    --------
    tuple
        (P(X < pi_l_than), P(pi_between[0] < X < pi_between[1]), P(X > pi_h_than))
        Returns None for any probability where the corresponding input is None
    """

    # Validate pi_between if provided
    if pi_between is not None:
        if len(pi_between) != 2:
            raise ValueError("pi_between must be an array of exactly 2 values")

    # Calculate requested probabilities
    p_less = (
        stats.gamma.cdf(pi_l_than, shape, scale=1 / rate)
        if pi_l_than is not None
        else None
    )

    p_between = (
        (
            stats.gamma.cdf(pi_between[1], shape, scale=1 / rate)
            - stats.gamma.cdf(pi_between[0], shape, scale=1 / rate)
        )
        if pi_between is not None
        else None
    )

    p_higher = (
        (1 - stats.gamma.cdf(pi_h_than, shape, scale=1 / rate))
        if pi_h_than is not None
        else None
    )

    return p_less, p_between, p_higher


def proba_normal(
    mean: float,
    sd: float,
    pi_l_than: float = None,
    pi_between: np.ndarray = None,
    pi_h_than: float = None,
):
    """
    Calculate probabilities for different regions of a Normal distribution.

    Parameters:
    -----------
    mean : float
        Mean of the Normal distribution
    sd : float
        Standard deviation of the Normal distribution
    pi_l_than : float, optional
        Upper bound for "less than" probability
    pi_between : np.ndarray, optional
        Array of [lower, upper] bounds for "between" probability
    pi_h_than : float, optional
        Lower bound for "higher than" probability

    Returns:
    --------
    tuple
        (P(X < pi_l_than), P(pi_between[0] < X < pi_between[1]), P(X > pi_h_than))
        Returns None for any probability where the corresponding input is None
    """

    # Validate pi_between if provided
    if pi_between is not None:
        if len(pi_between) != 2:
            raise ValueError("pi_between must be an array of exactly 2 values")

    # Calculate requested probabilities
    p_less = (
        stats.norm.cdf(pi_l_than, loc=mean, scale=sd) if pi_l_than is not None else None
    )

    p_between = (
        (
            stats.norm.cdf(pi_between[1], loc=mean, scale=sd)
            - stats.norm.cdf(pi_between[0], loc=mean, scale=sd)
        )
        if pi_between is not None
        else None
    )

    p_higher = (
        (1 - stats.norm.cdf(pi_h_than, loc=mean, scale=sd))
        if pi_h_than is not None
        else None
    )

    return p_less, p_between, p_higher


def quantiles_mcmc(
    vars_n_chains: pd.DataFrame, variable: str, quantiles: np.ndarray
) -> np.ndarray:
    """
    Calculate empirical quantiles for a given variable from MCMC samples.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from MCMC sampling with columns:
          - chain_{i} for each chain
          - sample_nr (index of draws within a chain)
          - var_name (the parameter name)
          - is_warmup (boolean indicating warmup vs. post-warmup draw), etc.
    variable : str
        Name of the parameter (var_name) to extract from the DataFrame.
    quantiles : np.ndarray
        Array of probabilities for which to compute empirical quantiles (e.g. [0.025, 0.5, 0.975]).

    Returns
    -------
    np.ndarray
        Quantile values corresponding to the requested probabilities, computed from the MCMC samples.
    """

    # 1. Filter rows only for the requested variable
    var_data = vars_n_chains[vars_n_chains["var_name"] == variable]

    # 2. Identify chain columns
    chain_cols = [col for col in var_data.columns if col.startswith("chain_")]

    # 3. Flatten all chain samples into a single 1D array
    samples = var_data[chain_cols].to_numpy().flatten()

    # 4. Compute quantiles (empirical) from these samples
    return np.quantile(samples, quantiles)


def proba_mcmc(
    vars_n_chains: pd.DataFrame,
    variable: str,
    pi_l_than: float = None,
    pi_between: np.ndarray = None,
    pi_h_than: float = None,
    filter_warmup: bool = True,
):
    """
    Calculate probabilities for different regions of an MCMC-derived distribution.

    This mimics proba_beta but uses empirical samples instead of a Beta distribution.

    Parameters
    ----------
    vars_n_chains : pd.DataFrame
        DataFrame from MCMC sampling with columns:
          - chain_{i} for each chain
          - sample_nr (index of draws within a chain)
          - var_name (the parameter name)
          - is_warmup (boolean indicating warmup vs. post-warmup draw)
    variable : str
        Name of the parameter (var_name) in vars_n_chains to analyze.
    pi_l_than : float, optional
        Upper bound for "less than" probability
    pi_between : np.ndarray, optional
        Array of [lower, upper] bounds for "between" probability
    pi_h_than : float, optional
        Lower bound for "higher than" probability
    filter_warmup : bool, optional
        If True, exclude warmup samples from the computation (default True)

    Returns
    -------
    tuple
        (P(X < pi_l_than), P(pi_between[0] < X < pi_between[1]), P(X > pi_h_than))
        Each probability is None if the corresponding input is None.
    """

    # 1. (Optional) Filter out warmup samples
    if filter_warmup and "is_warmup" in vars_n_chains.columns:
        df = vars_n_chains[~vars_n_chains["is_warmup"]]
    else:
        df = vars_n_chains

    # 2. Filter for the given variable
    df_var = df[df["var_name"] == variable]

    # 3. Flatten all chain_ columns into a single 1D array
    chain_cols = [col for col in df_var.columns if col.startswith("chain_")]
    samples = df_var[chain_cols].to_numpy().flatten()

    # Validate pi_between if provided
    if pi_between is not None:
        if len(pi_between) != 2:
            raise ValueError("pi_between must be an array of exactly 2 values")

    # 4. Calculate the probabilities
    p_less = None
    if pi_l_than is not None:
        p_less = np.mean(samples < pi_l_than)

    p_between = None
    if pi_between is not None:
        p_between = np.mean((samples > pi_between[0]) & (samples < pi_between[1]))

    p_higher = None
    if pi_h_than is not None:
        p_higher = np.mean(samples > pi_h_than)

    return p_less, p_between, p_higher


## posterior hypothesis testing
def odds(p_Ha: float):
    return p_Ha / (1 - p_Ha)


def bayes_factor(odds_posterior: float, odds_prior: float):
    return odds_posterior / odds_prior


# Linear Regression
def fit_glm(
    formula: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    auto_scale: bool = True,
    center_predictors: bool = True,
    num_chains: int = 4,
    num_samples: int = 10000,
    num_warmup: int = 5000,
    save_warmup: bool = False,
    random_seed: int = 42,
    priors: Optional[Dict[str, bmb.Prior]] = None,
    return_all: bool = False,
) -> pd.DataFrame:
    """
    Fits a Generalized Linear Model (GLM) using Bambi, with the option to specify priors.

    Parameters:
    - formula (str): The model formula, e.g., 'y ~ x1 + x2'.
    - data (pd.DataFrame): The dataset to be used for modeling.
    - family (str): The family distribution, e.g., 'gaussian', 'bernoulli', etc.
    - auto_scale (bool): Whether to automatically scale the priors to be informative.
    - center_predictors (bool): Wether to center (subtracting the mean) the data -> This changes interpretation of the priors.
    - num_chains (int): Number of MCMC chains.
    - num_samples (int): Number of samples per chain.
    - num_warmup (int): Number of warmup (burn-in) samples per chain.
    - save_warmup (bool): Whether to include warmup samples in the output.
    - random_seed (int): Random seed for reproducibility.
    - priors (Optional[Dict[str, bmb.Prior]]): A dictionary specifying priors for the model parameters.
    - return_all (bool): If True -> also returns fit object and model itself

    Returns:
    - pd.DataFrame: A concatenated DataFrame containing the sampling results for all parameters,
                    including information about the chain, sample number, variable name, and
                    whether the sample is part of the warmup.
    - fit: Fit object if return_all True
    - model: Fit model if return_all True
    """

    # Create the model, including priors if provided
    model = bmb.Model(
        formula=formula,
        data=data,
        family=family,
        auto_scale=auto_scale,
        center_predictors=center_predictors,
        priors=priors,
    )
    model.build()

    # Fit the model using MCMC sampling
    fit = model.fit(
        draws=num_samples,
        tune=num_warmup,
        chains=num_chains,
        discard_tuned_samples=not save_warmup,
        random_seed=random_seed,
    )

    # Convert the fit results to a DataFrame
    df = fit.to_dataframe()

    # Ensure all column names are tuples for MultiIndex compatibility
    df.columns = [col if isinstance(col, tuple) else (col, "") for col in df.columns]
    multi_index = pd.MultiIndex.from_tuples(df.columns)

    # Assign the MultiIndex to the DataFrame columns
    df.columns = multi_index

    if save_warmup:
        # Extract warmup samples from the DataFrame
        df_warmup = df[["chain", "draw", "warmup_posterior"]].copy()
        df_warmup = df_warmup.dropna()
        df_warmup = df_warmup.rename(columns={"warmup_posterior": "posterior"}, level=0)

        # Adjust the sample number for warmup samples
        df_warmup["draw"] -= num_warmup

    # Select relevant columns: chain, draw, and posterior samples
    df = df[["chain", "draw", "posterior"]]

    if save_warmup:
        # Concatenate warmup and posterior samples
        df = pd.concat([df_warmup, df], ignore_index=True)

    # Flatten the MultiIndex columns
    df.columns = [col[0] if len(col[1]) == 0 else col[1] for col in df.columns]

    # Transform the DataFrame to a long format with 'var_name' as a variable
    df = df.melt(id_vars=["chain", "draw"], var_name="var_name")

    # Pivot the DataFrame to have chains as separate columns
    df = df.pivot(index=["var_name", "draw"], columns="chain").reset_index()

    # Rename the chain columns for clarity
    df.columns = [
        col[0] if len(str(col[1])) == 0 else f"chain_{col[1]}" for col in df.columns
    ]

    # Rename the 'draw' column to 'sample_nr' for consistency
    df = df.rename(columns={"draw": "sample_nr"})

    # Create a flag to indicate whether a sample is part of the warmup
    df["is_warmup"] = df.sample_nr < 0

    # if return all enabled -> return df and fit
    if return_all:
        return df, fit, model

    return df

"""
Program to simulate the total losses of a loan portfolio using a Monte Carlo simulation.
There is the assumption of a large homogeneous porfolio.  
Default correlation is assumed to be constant and modelled with a common factor.
"""

import numpy as np
import random
import os
import sys
from scipy.stats import norm, t
import time
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

toll_zero_corr = 1e-6


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self): # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def set_seed(seed):
    """
    Sets the random seed for all packages that use random number generation and for os environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def simulate_default_times(n_loans, annual_pd, default_corr, n_periods, n_scenarios, dist_type='normal', df=None):
    """
    Simulates the time until default for each loan in a portfolio.
    
    Parameters:
    - n_loans: Number of loans in the portfolio.
    - annual_pd: Annual probability of default.
    - default_corr: Correlation between defaults in the portfolio.
    - n_periods: Number of periods to simulate.
    - n_scenarios: Number of Monte Carlo scenarios.
    - dist_type: Distribution type for simulation ('normal' or 't').
    - df: Degrees of freedom for the t-distribution (used only if dist_type is 't').
    """
    # Convert annual probability of default to monthly and compute cumulative PDs.
    monthly_pd = 1 - (1 - annual_pd) ** (1/12)
    cumulative_pd = np.array([1 - (1 - monthly_pd) ** i for i in range(1, n_periods + 1)])
    
    # Generate common and idiosyncratic factors based on the specified distribution.
    print("Creating common and idiosyncratic factors...")
    tick = time.time()
    idiosyncratic_factor = np.random.normal(0, 1, (n_loans, n_scenarios)) if dist_type == 'normal' else t.rvs(df, size=(n_loans, n_scenarios))
    if default_corr > toll_zero_corr:
        common_factor = np.random.normal(0, 1, (1, n_scenarios)) if dist_type == 'normal' else t.rvs(df, size=(1, n_scenarios))
        z = np.sqrt(default_corr) * common_factor + np.sqrt(1 - default_corr) * idiosyncratic_factor
    else:
        z = idiosyncratic_factor
    tock = time.time()
    print(f"Time taken: {tock - tick:.2f} seconds")
    
    # Determine default times by comparing generated factors with PD thresholds.
    print("Determining default times...")
    thresholds = norm.ppf(cumulative_pd) if dist_type == 'normal' else t.ppf(cumulative_pd, df)
    default_times = np.searchsorted(thresholds, z, side='right')
    default_times[default_times >= n_periods] = n_periods  # Adjust for non-defaulting loans.
    tock2 = time.time()
    print(f"Time taken: {tock2 - tock:.2f} seconds")
    return default_times

def calculate_losses(default_times, lgd, initial_balance, payment, n_periods, n_loans):
    """
    Calculates total losses for the loan portfolio based on default times.
    
    Parameters:
    - default_times: Array of default times for each loan and scenario.
    - lgd: Loss given default as a percentage of the balance at default.
    - initial_balance: Initial balance of each loan.
    - payment: Payment made each period.
    - n_periods: Number of periods.
    - n_loans: Number of loans.
    """
    # Calculate outstanding balance at each period.
    outstanding_balances = initial_balance - np.arange(n_periods) * payment
    
    # Calculate losses
    print("Calculating losses...")
    
    # Using numpy broadcasting to avoid loops.
    tick = time.time()
    losses = np.where(default_times < n_periods, initial_balance - default_times * payment, 0) * lgd
    # Aggregate losses across all loans.
    total_losses = losses.sum(axis=0)
    tock = time.time()
    print(f"Time taken: {tock - tick:.2f} seconds")
    
    return total_losses

def simulate_complete_portfolio(n_loans, annual_pd, default_corr, n_periods, n_scenarios, lgd, initial_balance, payment, dist_type, df=None):
    """
    Simulates the complete loan portfolio and calculates total losses.
    
    Parameters:
    - n_loans, annual_pd, default_corr, n_periods, n_scenarios, lgd, initial_balance, payment: Same as above.
    - dist_type, df: Distribution type and degrees of freedom for t-distribution.
    """
    # Simulate default times and calculate total losses.
    total_losses = simulate_default_times(n_loans, annual_pd, default_corr, n_periods, n_scenarios, dist_type, df)
    return calculate_losses(total_losses, lgd, initial_balance, payment, n_periods, n_loans)

def plot_cumulative_distribution(total_losses, total_initial_balance, quantile_levels, output_file):
    """
    Plots the cumulative distribution of total losses as a percentage of the initial portfolio balance.
    
    Parameters:
    - total_losses: Array of total losses for each scenario.
    - total_initial_balance: Total initial balance of the loan portfolio.
    - quantile_levels: Quantile levels to mark on the distribution plot.
    - output_file: File to save the plot to.
    """
    # Calculate and plot the distribution.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(total_losses / total_initial_balance * 100, kde=True, stat='density', bins=100, color='blue')
    
    # Calculate and mark quantiles and the mean on the plot.
    quantiles = np.quantile(total_losses, quantile_levels) / total_initial_balance * 100
    mean = np.mean(total_losses) / total_initial_balance * 100
    plt.axvline(mean, color='green', linestyle='--', label=f'Mean: {mean:.2f}%')
    print("Loss Results:")
    print(f"Mean: {mean:.3f}%")
    for i, quantile in enumerate(quantiles):
        plt.axvline(quantile, color='red', linestyle='--', label=f'{quantile_levels[i]*100:.2f}% Quantile: {quantile:.2f}%')
        # Print values on the console.
        print(f"{quantile_levels[i]*100:.3f}% Quantile: {quantile:.3f}%")
    plt.legend()
    plt.title('Distribution of Total Losses [% of Initial Balance]')
    plt.ylabel('Probability')
    plt.savefig(output_file)
    plt.show()


def main():
    # Define simulation parameters.
    n_loans = 10000
    annual_pd = 0.006
    default_corr = 0.05
    n_periods = 60
    n_scenarios = 20000
    lgd = 0.47
    initial_balance = 10000  # Initial balance per loan.

    payment = initial_balance / n_periods
    quantile_levels = [0.99, 0.999, 0.9999]
    dist_type = 'normal'  # Choose 'normal' or 't' for distribution type.
    df = None  # Degrees of freedom for t-distribution, used if dist_type is 't'.

    output_file_log = "output.txt"
    output_file_fig = "distrib_cum_loss_end.png"

    # Set random seed for reproducibility.
    random_seed = 12345
    set_seed(random_seed)

    #Determine operating system and clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    #Save log of all output to file and print to console
    sys.stdout = Logger(output_file_log)
    
    # Print simulation parameters summary.
    print("Simulation Parameters Summary:")
    print(f"Number of Loans: {n_loans:,}")
    print(f"Annual PD: {annual_pd*100:.2f}%")
    print(f"Default Correlation: {default_corr*100:.2f}%")
    print(f"Number of Periods: {n_periods:,}")
    print(f"Number of Scenarios: {n_scenarios:,}")
    print(f"LGD: {lgd*100:.2f}%")
    print(f"Initial Balance per Loan: {initial_balance:,.2f}")
    print(f"Total Initial Portfolio Balance: {initial_balance*n_loans:,.2f}")
    print(f"Distribution Type: {dist_type.capitalize()}")
    if dist_type == 't':
        print(f"Degrees of Freedom: {df}")
    print("")

    print("Running simulation...\n")
    start_time = time.time()
    total_losses = simulate_complete_portfolio(n_loans, annual_pd, default_corr, n_periods, n_scenarios, lgd, initial_balance, payment, dist_type, df)
    end_time = time.time()
    print("Simulation complete.")
    print(f"Total Time: {end_time - start_time:.2f} seconds\n")
    
    total_initial_balance = initial_balance * n_loans
    
    # Plot the distribution of total losses as a percentage of the initial portfolio balance.
    plot_cumulative_distribution(total_losses, total_initial_balance, quantile_levels, output_file_fig)


if __name__ == "__main__":
    main()

###############################################################################
# Portfolio Optimization Module
#
# This module implements Modern Portfolio Theory (MPT) to create optimized
# investment portfolios based on different risk tolerances. It uses historical
# price data to generate portfolios along the efficient frontier, maximizing
# expected return for given levels of risk.
#
# Key Features:
# - Generates low, medium, and high risk portfolios
# - Calculates efficient frontier points for visualization
# - Provides detailed performance metrics
# - Supports custom risk-free rates
#
# Dependencies:
# - pandas: Data manipulation and analysis
# - numpy: Numerical computations
# - pypfopt: Portfolio optimization algorithms
###############################################################################

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

###############################################################################
# Data class to hold portfolio metrics in a type-safe manner.
# This ensures consistent access to performance measurements across the module.
###############################################################################

@dataclass
class PortfolioMetrics:
    """
    Stores key performance metrics for a portfolio:
    - expected_annual_return: Predicted yearly return based on historical data
    - annual_volatility: Standard deviation of yearly returns
    - sharpe_ratio: Risk-adjusted return metric (excess return per unit of risk)
    """
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float

###############################################################################
# Main Portfolio Optimizer Class
#
# This class handles all portfolio optimization operations including:
# 1. Processing historical price data
# 2. Calculating returns and covariance matrices
# 3. Generating optimized portfolios for different risk levels
# 4. Computing efficient frontier points
###############################################################################

class RiskBasedPortfolioOptimizer:
    """
    Portfolio optimizer implementing Modern Portfolio Theory (MPT) strategies.
    Generates optimized portfolios based on historical price data and risk preferences.
    """
    
    ###########################################################################
    # Constructor: Initializes the optimizer with required data and parameters
    ###########################################################################

    def __init__(self, prices_df: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer with historical data and parameters.

        Parameters
        ----------
        prices_df : pandas.DataFrame
            Historical price data where:
            - Columns represent different assets/securities
            - Index contains dates in ascending order
            - Values are adjusted closing prices
            - No missing or null values allowed
            Example:
                        AAPL    GOOGL   MSFT
            2020-01-01  100.0   200.0   150.0
            2020-01-02  101.0   201.0   151.0

        risk_free_rate : float, optional (default=0.02)
            Annual risk-free rate used in Sharpe ratio calculations
            Example: 0.02 represents a 2% risk-free rate

        Important Notes
        --------------
        1. Price data should be pre-cleaned (no missing values)
        2. Dates should be in ascending order
        3. Minimum recommended history is 1 year of data
        4. Prices should be in consistent currency
        """
        # Store input parameters
        self.prices = prices_df
        self.risk_free_rate = risk_free_rate
        
        ###################################################################
        # Calculate core matrices used in optimization
        # These are computed once at initialization to improve performance
        ###################################################################

        # Calculate expected returns using historical mean return method
        # This assumes past returns are indicative of future performance
        self.returns = expected_returns.mean_historical_return(prices_df)
        
        # Calculate the sample covariance matrix of asset returns
        # This measures how assets move together and is crucial for risk assessment
        self.cov_matrix = risk_models.sample_cov(prices_df)
        
        # Initialize cache for efficient frontier points
        # This prevents redundant calculations when generating frontier multiple times
        self._efficient_frontier_cache: Optional[List[Tuple]] = None
    
    ###########################################################################
    # Private Helper Methods
    ###########################################################################

    def _create_efficient_frontier(self, **kwargs) -> EfficientFrontier:
        """
        Creates a new EfficientFrontier instance with current data and parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to EfficientFrontier constructor

        Returns
        -------
        EfficientFrontier
            Configured optimizer instance ready for portfolio generation

        Notes
        -----
        1. Uses current returns and covariance matrix
        2. Enforces long-only positions (no short selling)
        3. Applies any additional constraints passed via kwargs
        """
        return EfficientFrontier(
            expected_returns=self.returns,
            cov_matrix=self.cov_matrix,
            weight_bounds=(0, 1),  # Enforce long-only positions
            **kwargs
        )
        
    def _get_portfolio_metrics(self, ef: EfficientFrontier) -> Dict:
        """
        Extracts performance metrics from an optimized portfolio.

        Parameters
        ----------
        ef : EfficientFrontier
            Optimized portfolio object

        Returns
        -------
        dict
            Dictionary containing:
            - expected_annual_return: Predicted yearly return
            - annual_volatility: Yearly standard deviation
            - sharpe_ratio: Risk-adjusted return metric

        Notes
        -----
        1. All metrics are annualized
        2. Return and volatility are expressed as decimals (not percentages)
        3. Uses the risk-free rate specified during initialization
        """
        expected_return, volatility, sharpe = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate)  # Explicitly pass the same risk-free rate
        
        return {
            'expected_annual_return': expected_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe}
    
    ###########################################################################
    # Public Methods for Portfolio Generation and Analysis
    ###########################################################################

    def generate_risk_based_portfolios(self) -> Dict[str, Dict]:
        """
        Generates three distinct portfolios optimized for different risk preferences.

        Returns
        -------
        dict
            Nested dictionary containing portfolio data:
            {
                'low_risk': {
                    'weights': {asset: weight},
                    'metrics': {metric: value}
                },
                'medium_risk': {...},
                'high_risk': {...}
            }

        Portfolio Types
        --------------
        1. Low Risk:
           - Minimizes portfolio volatility
           - Best for risk-averse investors
           - May sacrifice returns for stability

        2. Medium Risk:
           - Maximizes Sharpe ratio
           - Optimal risk-adjusted returns
           - Balanced approach to risk/return

        3. High Risk:
           - Targets 50% higher volatility than medium risk
           - Aims for higher returns
           - Suitable for risk-tolerant investors

        Notes
        -----
        1. All portfolios are long-only (no short selling)
        2. Weights sum to 100% for each portfolio
        3. Uses mean historical returns for expected returns
        """
        portfolios = {}
        
        ###################################################################
        # Generate Low Risk Portfolio
        # Strategy: Minimize volatility regardless of returns
        ###################################################################
        
        ef_low = self._create_efficient_frontier()
        ef_low.min_volatility()
        portfolios['low_risk'] = {
            'weights': ef_low.clean_weights(),
            'metrics': self._get_portfolio_metrics(ef_low)
        }
        
        ###################################################################
        # Generate Medium Risk Portfolio
        # Strategy: Maximize Sharpe ratio (risk-adjusted returns)
        ###################################################################
        
        ef_medium = self._create_efficient_frontier()
        ef_medium.max_sharpe(risk_free_rate=self.risk_free_rate)
        _, baseline_vol, _ = ef_medium.portfolio_performance()
        portfolios['medium_risk'] = {
            'weights': ef_medium.clean_weights(),
            'metrics': self._get_portfolio_metrics(ef_medium)
        }
        
        ###################################################################
        # Generate High Risk Portfolio
        # Strategy: Target higher volatility for potentially higher returns
        ###################################################################
        
        ef_high = self._create_efficient_frontier()
        # Set target volatility 50% higher than the medium risk portfolio
        target_vol = baseline_vol * 1.5
        ef_high.efficient_risk(target_vol)
        portfolios['high_risk'] = {
            'weights': ef_high.clean_weights(),
            'metrics': self._get_portfolio_metrics(ef_high)
        }
        
        return portfolios
    
    def generate_efficient_frontier_points(self, points: int = 50) -> List[Tuple]:
        """
        Generates points along the efficient frontier for visualization.

        Parameters
        ----------
        points : int, optional (default=50)
            Number of points to generate along the frontier
            More points = smoother curve but slower computation

        Returns
        -------
        List[Tuple]
            List of (volatility, return) pairs representing efficient portfolios,
            sorted by volatility in ascending order

        Implementation Details
        --------------------
        1. Finds minimum volatility portfolio
        2. Finds maximum Sharpe ratio portfolio
        3. Generates evenly spaced volatility targets between min and max * 1.5
        4. Creates optimal portfolio for each volatility target
        5. Caches results to avoid redundant calculations

        Notes
        -----
        1. Uses caching to improve performance on repeated calls
        2. May return fewer points than requested if optimization fails
        3. Points are guaranteed to be sorted by volatility
        """
        ###################################################################
        # Check Cache
        # Return cached results if available to avoid redundant calculations
        ###################################################################
        
        if self._efficient_frontier_cache is not None:
            return self._efficient_frontier_cache
            
        ###################################################################
        # Find Volatility Range
        # Determine the range of volatilities to consider
        ###################################################################
        
        # Find minimum volatility portfolio
        ef_min = self._create_efficient_frontier()
        ef_min.min_volatility()
        _, min_vol, _ = ef_min.portfolio_performance()
        
        # Find maximum Sharpe ratio portfolio
        ef_max = self._create_efficient_frontier()
        ef_max.max_sharpe(risk_free_rate=self.risk_free_rate)
        _, max_vol, _ = ef_max.portfolio_performance()
        
        ###################################################################
        # Generate Frontier Points
        # Create portfolios across the volatility range
        ###################################################################
        
        # Create array of target volatilities
        target_vols = np.linspace(min_vol, max_vol * 1.5, points)
        efficient_frontier_points = []
        
        # Generate optimal portfolio for each target volatility
        for target_vol in target_vols:
            ef = self._create_efficient_frontier()
            try:
                # Optimize portfolio for target volatility
                ef.efficient_risk(target_vol)
                ret, vol, _ = ef.portfolio_performance()
                efficient_frontier_points.append((vol, ret))
            except Exception:
                # Skip if optimization fails for this volatility target
                continue
        
        ###################################################################
        # Sort and Cache Results
        ###################################################################
        
        # Sort points by volatility for consistent output
        self._efficient_frontier_cache = sorted(efficient_frontier_points)
        return self._efficient_frontier_cache

###############################################################################
# Utility Functions
###############################################################################

def format_portfolio_report(portfolios: Dict, min_allocation: float = 0.01) -> str:
    """
    Formats portfolio analysis results into a human-readable report.

    Parameters
    ----------
    portfolios : dict
        Portfolio data from generate_risk_based_portfolios()
    min_allocation : float, optional (default=0.01)
        Minimum allocation threshold to display (1% = 0.01)

    Returns
    -------
    str
        Formatted report containing:
        - Portfolio risk level
        - Expected return, volatility, and Sharpe ratio
        - Asset allocations above minimum threshold

    Example Output
    -------------
    LOW RISK PORTFOLIO
    --------------------------------------------------
    Portfolio Metrics:
    Expected Annual Return: 8.50%
    Annual Volatility: 12.30%
    Sharpe Ratio: 0.53

    Asset Allocation:
    AAPL: 25.00%
    MSFT: 35.00%
    GOOGL: 40.00%

    Notes
    -----
    1. Filters out very small allocations for clarity
    2. All percentages are formatted with 2 decimal places
    3. Each portfolio section is clearly separated
    """
    report_sections = []
    
    ###########################################################################
    # Generate Report Sections
    # Create formatted section for each portfolio risk level
    ###########################################################################
    
    for risk_level, portfolio in portfolios.items():
        metrics = portfolio['metrics']
        weights = portfolio['weights']
        
        # Format section header and metrics
        section = [
            f"\n{risk_level.upper().replace('_', ' ')} PORTFOLIO",
            "-" * 50,
            "Portfolio Metrics:",
            f"Expected Annual Return: {metrics['expected_annual_return']:.2%}",
            f"Annual Volatility: {metrics['annual_volatility']:.2%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            "\nAsset Allocation:"
        ]
        
        ###################################################################
        # Add Asset Allocations
        # Include only allocations above minimum threshold
        ###################################################################
        
        allocations = [
            f"{asset}: {weight:.2%}"
            for asset, weight in weights.items()
            if weight > min_allocation
        ]
        section.extend(allocations)
        section.append("")  # Add blank line between sections
        
        report_sections.extend(section)
    
    ###########################################################################
    # Combine and Return
    # Join all sections with newlines for final report
    ###########################################################################
    
    return "\n".join(report_sections)

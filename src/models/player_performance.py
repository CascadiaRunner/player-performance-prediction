"""
Bayesian models for player performance analysis.
"""
import numpy as np
from scipy import stats
import pymc3 as pm
import arviz as az
from typing import Tuple, Dict, Optional

class PlayerPerformance:
    """
    Bayesian model for estimating player performance.
    """
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize the model with prior parameters.
        
        Args:
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posterior_alpha = prior_alpha
        self.posterior_beta = prior_beta
        
    def update_posterior(self, goals: int, minutes_played: int) -> None:
        """
        Update the posterior distribution with new data using conjugate prior.
        
        Args:
            goals: Number of goals scored
            minutes_played: Total minutes played
        """
        # Convert minutes to 90-minute matches
        matches = minutes_played / 90
        
        # Update posterior parameters (Beta-Binomial conjugate)
        self.posterior_alpha = self.prior_alpha + goals
        self.posterior_beta = self.prior_beta + (matches - goals)
        
    def get_credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Calculate credible interval for the posterior distribution.
        
        Args:
            alpha: Credible interval level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return stats.beta.ppf([(1 - alpha) / 2, 1 - (1 - alpha) / 2],
                             self.posterior_alpha,
                             self.posterior_beta)
    
    def get_posterior_mean(self) -> float:
        """
        Get the posterior mean (expected goals per match).
        
        Returns:
            Posterior mean
        """
        return self.posterior_alpha / (self.posterior_alpha + self.posterior_beta)
    
    def get_posterior_variance(self) -> float:
        """
        Get the posterior variance.
        
        Returns:
            Posterior variance
        """
        total = self.posterior_alpha + self.posterior_beta
        return (self.posterior_alpha * self.posterior_beta) / (total ** 2 * (total + 1))

class HierarchicalPlayerPerformance:
    """
    Hierarchical Bayesian model for player performance using MCMC.
    """
    def __init__(self, league_mean: float, league_std: float):
        """
        Initialize the hierarchical model.
        
        Args:
            league_mean: League-wide mean goals per 90
            league_std: League-wide standard deviation of goals per 90
        """
        self.league_mean = league_mean
        self.league_std = league_std
        self.trace = None
        
    def fit(self, player_data: Dict[str, np.ndarray],
            draws: int = 2000,
            tune: int = 1000) -> None:
        """
        Fit the hierarchical model using MCMC.
        
        Args:
            player_data: Dictionary containing:
                - 'player_ids': Array of player IDs
                - 'goals': Array of goals scored
                - 'minutes': Array of minutes played
            draws: Number of posterior samples
            tune: Number of tuning steps
        """
        with pm.Model() as model:
            # Hyperpriors
            mu = pm.Normal('mu', mu=self.league_mean, sigma=self.league_std)
            sigma = pm.HalfNormal('sigma', sigma=self.league_std)
            
            # Player-specific parameters
            n_players = len(np.unique(player_data['player_ids']))
            theta = pm.Normal('theta', mu=mu, sigma=sigma, shape=n_players)
            
            # Likelihood
            goals = pm.Poisson('goals',
                             mu=theta[player_data['player_ids']] * player_data['minutes'] / 90,
                             observed=player_data['goals'])
            
            # Sample from posterior
            self.trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    def get_player_estimates(self, player_id: int) -> Dict[str, float]:
        """
        Get posterior estimates for a specific player.
        
        Args:
            player_id: Player ID
            
        Returns:
            Dictionary containing:
                - 'mean': Posterior mean
                - 'std': Posterior standard deviation
                - 'ci_lower': Lower 95% credible interval
                - 'ci_upper': Upper 95% credible interval
        """
        if self.trace is None:
            raise ValueError("Model must be fit before getting estimates")
            
        player_samples = self.trace.posterior.theta[:, :, player_id].values.flatten()
        
        return {
            'mean': np.mean(player_samples),
            'std': np.std(player_samples),
            'ci_lower': np.percentile(player_samples, 2.5),
            'ci_upper': np.percentile(player_samples, 97.5)
        }
    
    def plot_posterior(self, player_id: int, save_path: Optional[str] = None) -> None:
        """
        Plot posterior distribution for a player.
        
        Args:
            player_id: Player ID
            save_path: Optional path to save the plot
        """
        if self.trace is None:
            raise ValueError("Model must be fit before plotting")
            
        az.plot_posterior(self.trace, var_names=['theta'],
                         coords={'theta_dim_0': player_id},
                         save_path=save_path) 
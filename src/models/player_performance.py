"""
Bayesian models for player performance analysis.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt

class PlayerPerformance:
    """
    Bayesian model for estimating player performance.
    """
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, min_minutes: int = 540):
        """
        Initialize the model with prior parameters.
        
        Args:
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
            min_minutes: Minimum minutes required for analysis (default: 540 = 6 full matches)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posterior_alpha = prior_alpha
        self.posterior_beta = prior_beta
        self.min_minutes = min_minutes
        self.minutes_played = 0
        
    def update_posterior(self, goals: int, minutes_played: int) -> None:
        """
        Update the posterior distribution with new data using conjugate prior.
        
        Args:
            goals: Number of goals scored
            minutes_played: Total minutes played
        """
        self.minutes_played = minutes_played
        
        # Convert minutes to 90-minute matches
        matches = minutes_played / 90
        
        # Update posterior parameters (Beta-Binomial conjugate)
        self.posterior_alpha = self.prior_alpha + goals
        self.posterior_beta = self.prior_beta + (matches - goals)
        
    def has_sufficient_minutes(self) -> bool:
        """
        Check if the player has played enough minutes for reliable analysis.
        
        Returns:
            bool: True if player has sufficient minutes, False otherwise
        """
        return self.minutes_played >= self.min_minutes
        
    def get_credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Calculate credible interval for the posterior distribution.
        
        Args:
            alpha: Credible interval level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.has_sufficient_minutes():
            return (np.nan, np.nan)
            
        return stats.beta.ppf([(1 - alpha) / 2, 1 - (1 - alpha) / 2],
                             self.posterior_alpha,
                             self.posterior_beta)
    
    def get_posterior_mean(self) -> float:
        """
        Get the posterior mean (expected goals per match).
        
        Returns:
            Posterior mean
        """
        if not self.has_sufficient_minutes():
            return np.nan
            
        return self.posterior_alpha / (self.posterior_alpha + self.posterior_beta)
    
    def get_posterior_variance(self) -> float:
        """
        Get the posterior variance.
        
        Returns:
            Posterior variance
        """
        if not self.has_sufficient_minutes():
            return np.nan
            
        total = self.posterior_alpha + self.posterior_beta
        if total <= 0:
            return np.nan
            
        variance = (self.posterior_alpha * self.posterior_beta) / (total ** 2 * (total + 1))
        return max(variance, 1e-10)  # Ensure minimum positive variance
    
    def plot_posterior(self, save_path: Optional[str] = None) -> None:
        """
        Plot the posterior distribution.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.has_sufficient_minutes():
            raise ValueError("Insufficient minutes played for reliable analysis")
            
        x = np.linspace(0, 1, 1000)
        y = stats.beta.pdf(x, self.posterior_alpha, self.posterior_beta)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', lw=2, label='Posterior')
        
        # Add prior distribution
        y_prior = stats.beta.pdf(x, self.prior_alpha, self.prior_beta)
        plt.plot(x, y_prior, 'r--', lw=2, label='Prior')
        
        plt.fill_between(x, y, alpha=0.2)
        plt.xlabel('Goals per Match')
        plt.ylabel('Density')
        plt.title('Posterior Distribution of Goal-Scoring Rate')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

class HierarchicalPlayerPerformance:
    """
    Hierarchical Bayesian model for player performance using empirical Bayes.
    """
    def __init__(self, league_mean: float, league_std: float, min_minutes: int = 540):
        """
        Initialize the hierarchical model.
        
        Args:
            league_mean: League-wide mean goals per 90
            league_std: League-wide standard deviation of goals per 90
            min_minutes: Minimum minutes required for analysis (default: 540 = 6 full matches)
        """
        self.league_mean = league_mean
        self.league_std = league_std
        self.min_minutes = min_minutes
        self.player_models = {}
        self.player_history = {}  # Store player's historical performance
        
        # Position-specific prior adjustments
        self.position_priors = {
            'forward': {'mean_multiplier': 1.5, 'std_multiplier': 1.2},
            'midfielder': {'mean_multiplier': 0.8, 'std_multiplier': 1.1},
            'defender': {'mean_multiplier': 0.4, 'std_multiplier': 1.3}
        }
        
    def _infer_position(self, goals_per_90: float) -> str:
        """
        Infer player position based on goal-scoring rate.
        
        Args:
            goals_per_90: Player's goals per 90 minutes
            
        Returns:
            Inferred position ('forward', 'midfielder', or 'defender')
        """
        if goals_per_90 > 0.5:  # High scoring rate suggests forward
            return 'forward'
        elif goals_per_90 > 0.2:  # Moderate scoring rate suggests midfielder
            return 'midfielder'
        else:  # Low scoring rate suggests defender
            return 'defender'
    
    def _calculate_development_factor(self, player_id: int, current_goals: float, current_minutes: float) -> float:
        """
        Calculate player development factor based on historical performance.
        
        Args:
            player_id: Player ID
            current_goals: Current season goals
            current_minutes: Current season minutes
            
        Returns:
            Development factor (1.0 means no change, >1.0 means improvement, <1.0 means decline)
        """
        if player_id not in self.player_history:
            return 1.0  # No history available
            
        historical_goals_per_90 = self.player_history[player_id]['goals_per_90']
        current_goals_per_90 = (current_goals * 90) / current_minutes
        
        # Calculate development factor based on performance change
        if historical_goals_per_90 == 0:
            return 1.0 if current_goals_per_90 == 0 else 1.2  # Assume improvement if scoring for first time
            
        development_factor = current_goals_per_90 / historical_goals_per_90
        
        # Cap the development factor to reasonable bounds
        return max(0.5, min(2.0, development_factor))
    
    def fit(self, player_data: Dict[str, np.ndarray]) -> None:
        """
        Fit the hierarchical model using empirical Bayes.
        
        Args:
            player_data: Dictionary containing:
                - 'player_ids': Array of player IDs
                - 'goals': Array of goals scored
                - 'minutes': Array of minutes played
        """
        unique_players = np.unique(player_data['player_ids'])
        
        # First pass: calculate historical performance
        for player_id in unique_players:
            mask = player_data['player_ids'] == player_id
            goals = player_data['goals'][mask]
            minutes = player_data['minutes'][mask]
            
            if minutes.sum() >= self.min_minutes:
                goals_per_90 = (goals.sum() * 90) / minutes.sum()
                self.player_history[player_id] = {
                    'goals_per_90': goals_per_90,
                    'minutes': minutes.sum()
                }
        
        # Second pass: fit models with position-specific priors
        for player_id in unique_players:
            mask = player_data['player_ids'] == player_id
            goals = player_data['goals'][mask]
            minutes = player_data['minutes'][mask]
            
            # Skip players with insufficient minutes
            if minutes.sum() < self.min_minutes:
                continue
            
            # Calculate goals per 90 for position inference
            goals_per_90 = (goals.sum() * 90) / minutes.sum()
            position = self._infer_position(goals_per_90)
            
            # Get position-specific prior adjustments
            position_prior = self.position_priors[position]
            adjusted_mean = self.league_mean * position_prior['mean_multiplier']
            adjusted_std = self.league_std * position_prior['std_multiplier']
            
            # Calculate development factor
            dev_factor = self._calculate_development_factor(player_id, goals.sum(), minutes.sum())
            
            # Calculate base prior parameters
            # Ensure mean is between 0 and 1 for Beta distribution
            adjusted_mean = max(0.01, min(0.99, adjusted_mean))
            
            # Calculate alpha and beta ensuring they're positive
            alpha = (adjusted_mean ** 2) * (1 - adjusted_mean) / (adjusted_std ** 2)
            beta = alpha * (1 / adjusted_mean - 1)
            
            # Apply development factor while maintaining valid Beta parameters
            # Scale both parameters proportionally to maintain the mean
            scale_factor = dev_factor
            alpha = max(0.1, alpha * scale_factor)
            beta = max(0.1, beta * (2 - scale_factor))
            
            # Create and fit player model
            model = PlayerPerformance(prior_alpha=alpha, prior_beta=beta, min_minutes=self.min_minutes)
            model.update_posterior(goals.sum(), minutes.sum())
            
            self.player_models[player_id] = model
    
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
        if player_id not in self.player_models:
            raise ValueError(f"No model found for player {player_id}")
            
        model = self.player_models[player_id]
        if not model.has_sufficient_minutes():
            return {
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
            
        ci_lower, ci_upper = model.get_credible_interval()
        
        return {
            'mean': model.get_posterior_mean(),
            'std': np.sqrt(model.get_posterior_variance()),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def get_qualified_players(self) -> List[int]:
        """
        Get list of player IDs who have sufficient minutes played.
        
        Returns:
            List of player IDs
        """
        return [pid for pid, model in self.player_models.items() 
                if model.has_sufficient_minutes()]
    
    def plot_posterior(self, player_id: int, save_path: Optional[str] = None) -> None:
        """
        Plot posterior distribution for a player.
        
        Args:
            player_id: Player ID
            save_path: Optional path to save the plot
        """
        if player_id not in self.player_models:
            raise ValueError(f"No model found for player {player_id}")
            
        self.player_models[player_id].plot_posterior(save_path) 
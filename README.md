# Player Performance Soccer

A Python project for analyzing soccer player performance using Bayesian statistics.

## Project Overview

This project demonstrates the application of Bayesian statistics to estimate player performance, particularly focusing on:
- Probability Theory (binomial models for success/failure)
- Inferential Statistics (MLE estimates)
- Bayesian Statistics (Bayesian updating of player performance)
- MCMC (sampling posterior distributions)

## Project Structure

```
player-performance-soccer/
├── data/
│   ├── raw/                 # Original data files
│   └── processed/           # Cleaned and processed data
├── src/
│   ├── data/               # Data processing scripts
│   ├── models/             # Statistical models
│   ├── visualization/      # Plotting scripts
│   └── utils/              # Helper functions
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
└── README.md              # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd player-performance-soccer
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Add usage instructions here]

## Statistical Methods

### Bayesian Approach
- Prior specification using historical data
- Posterior computation using conjugate priors
- MCMC sampling for complex models
- Credible intervals for uncertainty quantification

### Model Comparison
- Maximum Likelihood Estimation (MLE)
- Bayesian estimates with different priors
- Performance comparison in early-season scenarios

## Contributing

[Add contribution guidelines here]

## License

[Add license information here] 
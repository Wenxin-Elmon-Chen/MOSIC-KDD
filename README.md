# MOSIC: Model-agnostic Optimal Subgroup Identification with multi-Constraint

MOSIC is a machine learning framework for identifying optimal subgroups in causal inference settings with multiple constraints. The framework supports various identifier types (MLP, GradTree) and can handle constraints such as budget limits, safety requirements, and fairness considerations.

## Project Structure

```
MOSIC/
├── src/                           # Core implementation
│   ├── MOSIC.py                 # Main MOSIC algorithm
│   ├── GradTree.py               # Gradient tree identifier
│   ├── DecisionTree.py           # Decision tree utilities
│   ├── ps.py                     # Propensity score estimation
│   ├── eval_utils.py             # Evaluation utilities
│   ├── utils.py                  # General utilities
│   └── entmax.py                 # Entmax activation functions
├── experiments/                   # Experiment scripts
│   ├── synthetic_data_generation.py              # Generate synthetic datasets
│   ├── synthetic_data_train_nuisance_function.py # Generate IPW and DragonNet outputs
│   ├── synthetic_data_train_mosic_mlp.py        # MOSIC with MLP identifier
│   ├── synthetic_data_train_mosic_dt.py         # MOSIC with GradTree identifier
│   ├── synthetic_mosic_mlp_safety.py            # MOSIC with safety constraints
│   ├── synthetic_mosic_mlp_budget_safety.py     # MOSIC with budget + safety
│   └── synthetic_mosic_mlp_budget_safety_fairness.py # MOSIC with all constraints
├── data/                          # Generated datasets
├── main.py                        # Experiment runner
├── notebook/
|   └──MOSIC_Example.ipynb           # Jupyter notebook example
└── requirements.txt               # Python dependencies
```

## Setup

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MOSIC
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p results/GAMMA_0 results/GAMMA_5
mkdir -p data/syn-gamma-0 data/syn-gamma-5
```

## Usage

### Quick Start

**Option 1: Interactive Jupyter Notebook**
```bash
jupyter notebook notebook/MOSIC_Example.ipynb
```

**Option 2: Run all experiments with default settings**
```bash
python main.py
```

### Step-by-Step Workflow

#### 1. Generate Synthetic Data
```bash
python experiments/synthetic_data_generation.py
```
This creates:
- `data/syn-gamma-0/seed{1-100}.pkl` - Synthetic datasets with gamma=0
- `data/syn-gamma-5/seed{1-100}.pkl` - Synthetic datasets with gamma=5
- Corresponding scaler files for preprocessing

#### 2. Generate Nuisance Functions
```bash
python experiments/synthetic_data_train_nuisance_function.py
```
This generates:
- IPW (Inverse Propensity Weighting) estimates
- DragonNet outcome predictions
- Files: `seed{X}-ipw.pkl` and `seed{X}-dragonnet-output.pkl`

**Note**: You may need to modify this script to process both GAMMA values (0 and 5) and all seed ranges (1-100).

#### 3. Run MOSIC Experiments

Using the experiment runner:
```bash
# Run specific experiments
python main.py --experiments syn_mlp syn_dt

# Run with custom parameters
python main.py --experiments syn_mlp --gamma 5 --device cuda:0

# List all available experiments
python main.py --list
```

## Available Experiments

| Experiment | Datasets | Implementation | Constraints |
|------------|-------------|------------|-------------|
| `syn_mlp` | Confounded synthetic data | MOSIC-MLP | Group size, overlap |
| `syn_mlp_gamma0` | Unconfounded synthetic data | MOSIC-MLP | Group size, overlap |
| `syn_dt` | Confounded synthetic data | MOSIC-DT | Group size, overlap |
| `syn_dt_gamma0` | Unconfounded synthetic data| MOSIC-DT | Group size, overlap |
| `syn_mlp_safety` | Confounded synthetic data | MOSIC-MLP | + Safety risk |
| `syn_mlp_budget_safety` | Confounded synthetic data | MOSIC-MLP | + Budget, safety |
| `syn_mlp_budget_safety_fairness` | Confounded synthetic data | MOSIC-MLP | + Budget, safety, fairness |

## Configuration

## Key Features

### Identifiers
- **MLP**: Multi-layer perceptron for flexible subgroup identification
- **GradTree**: Gradient-based decision tree for interpretable rules

### Constraints
- **Group Size**: Control the proportion of population in identified subgroup
- **Overlap**: Minimize overlap between treatment and control groups
- **Budget**: Limit total cost
- **Safety**: Bound safety risk in identified subgroups  
- **Fairness**: Ensure equitable treatment across sensitive groups

### Evaluation Metrics
- **Ground Truth ATE**
- **ATE Estimation Error**: Estimation Error by AIPTW.
- **Covariate Balance**: Covariate balance measured by SMD (Standardized Mean Difference)
- **Additional Constraint Satisfaction**: Whether identified subgroups meet all constraints

## Examples

### Jupyter Notebook Tutorial
A comprehensive example is provided in `MOSIC_Example.ipynb` which demonstrates:
- Loading synthetic data with seed=1
- Setting up budget, safety, and fairness constraints
- Training a MOSIC model with MLP identifier
- Evaluating results and constraint satisfaction
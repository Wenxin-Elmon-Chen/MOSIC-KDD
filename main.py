import os
import argparse
import subprocess
import sys
from typing import Dict, Any, List

# Default experiment configurations
EXPERIMENT_CONFIGS = {
    "syn_mlp": {
        "script": "experiments/synthetic_data_train_mosic3_mlp.py",
        "description": "MOSIC3 with MLP identifier on synthetic data",
        "parameters": {
            "GAMMA": 5,
            "alphas": [0.0,0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "mlp",
            "hidden_size_list": [50, 100, 200],
            "beta_list": [1e-2,1e-3, 1e-4, 1e-5],
            "result_dir": "results",
            "seeds": list(range(1, 101)),  # Add more seeds as needed
            "expect_group_sizes": [0.3,0.4,0.5,0.6,0.7,0.8],
            "device": "cuda:0"
        }
    },
    "syn_mlp_gamma0": {
        "script": "experiments/synthetic_data_train_mosic3_mlp.py",
        "description": "MOSIC3 with MLP identifier on synthetic data with gamma=0",
        "parameters": {
            "GAMMA": 0,
            "alphas": [0.0,0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "mlp",
            "hidden_size_list": [50, 100, 200],
            "beta_list": [1e-2,1e-3, 1e-4, 1e-5],
            "result_dir": "results",
            "seeds": list(range(1, 101)),
            "expect_group_sizes": [0.3,0.4,0.5,0.6,0.7,0.8],
            "device": "cuda:1"
        }
    },
    "syn_dt": {
        "script": "experiments/synthetic_data_train_mosic3_dt.py",
        "description": "MOSIC3 with GradTree identifier on synthetic data",
        "parameters": {
            "GAMMA": 5,
            "alphas": [0.0,0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "dt",
            "depth_list": [3,5,7],
            "beta_list": [1e-5,1e-4,1e-3,1e-2],
            "result_dir": "results",
            "seeds": list(range(1, 101)),
            "expect_group_sizes": [0.3,0.4,0.5,0.6,0.7,0.8],
            "device": "cuda:1"
        }
    },
    "syn_dt_gamma0": {
        "script": "experiments/synthetic_data_train_mosic3_dt.py",
        "description": "MOSIC3 with GradTree identifier on synthetic data with gamma=0",
        "parameters": {
            "GAMMA": 0,
            "alphas": [0.0,0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "dt",
            "depth_list": [3,5,7],
            "beta_list": [1e-5,1e-4,1e-3,1e-2],
            "result_dir": "results",
            "seeds": list(range(1, 101)),
            "expect_group_sizes": [0.3,0.4,0.5,0.6,0.7,0.8],
            "device": "cuda:0"
        }
    },
    "syn_mlp_budget_safety_fairness": {
        "script": "experiments/synthetic_mosic3_mlp_budget_safety_fairness.py",
        "description": "MOSIC3 with MLP identifier on synthetic data with budget, safety and fairness constraints",
        "parameters": {
            "GAMMA": 5,
            "alphas": [0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "mlp",
            "hidden_size_list": [50, 100, 200],
            "beta_list": [1e-5, 1e-4, 1e-3, 1e-2],
            "result_dir": "results",
            "seeds": list(range(1, 101)),  # Add more seeds as needed
            "expect_group_sizes": [0.5],
            "device": "cuda:1"
        }
    },
    "syn_mlp_budget_safety": {
        "script": "experiments/synthetic_mosic3_mlp_budget_safety.py",
        "description": "MOSIC3 with MLP identifier on synthetic data with budget and safety constraints",
        "parameters": {
            "GAMMA": 5,
            "alphas": [0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "mlp",
            "hidden_size_list": [50, 100, 200],
            "beta_list": [1e-5, 1e-4, 1e-3, 1e-2],
            "result_dir": "results",
            "seeds": list(range(1, 101)),  # Add more seeds as needed
            "expect_group_sizes": [0.5],
            "device": "cpu"
        }
    },
    "syn_mlp_safety": {
        "script": "experiments/synthetic_mosic3_mlp_safety.py",
        "description": "MOSIC3 with MLP identifier on synthetic data with safety constraints",
        "parameters": {
            "GAMMA": 5,
            "alphas": [0.02],
            "L1_LAMBDA": 0.01,
            "identifier_type": "mlp",
            "hidden_size_list": [50, 100, 200],
            "beta_list": [1e-5, 1e-4, 1e-3, 1e-2],
            "result_dir": "results",
            "seeds": list(range(1, 101)),  # Add more seeds as needed
            "expect_group_sizes": [0.5],
            "device": "cpu"
        }
    },
}

def run_experiment(experiment_name: str, config: Dict[str, Any], 
                  override_params: Dict[str, Any] = None) -> None:
    """
    Run a single experiment with the given configuration.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        override_params: Parameters to override in the config
    """
    script_path = config["script"]
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found!")
        return
    
    # Merge default parameters with overrides
    params = config["parameters"].copy()
    if override_params:
        params.update(override_params)
    
    print(f"Running experiment: {experiment_name}")
    print(f"Script: {script_path}")
    print(f"Parameters: {params}")
    print("-" * 50)
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    
    # Add parameters as environment variables
    for key, value in params.items():
        if isinstance(value, list):
            env[f"PARAM_{key}"] = str(value).replace(" ", "")
        else:
            env[f"PARAM_{key}"] = str(value)
    
    # Run the experiment script
    try:
        result = subprocess.run([sys.executable, script_path], 
                              env=env, 
                              capture_output=False,
                              text=True)
        
        if result.returncode == 0:
            print(f"✅ Experiment {experiment_name} completed successfully!")
        else:
            print(f"❌ Experiment {experiment_name} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running experiment {experiment_name}: {str(e)}")

def run_all_experiments(experiments: List[str] = None, 
                       override_params: Dict[str, Any] = None) -> None:
    """
    Run multiple experiments.
    
    Args:
        experiments: List of experiment names to run. If None, runs all.
        override_params: Parameters to override for all experiments
    """
    if experiments is None:
        experiments = list(EXPERIMENT_CONFIGS.keys())
    
    print(f"Running {len(experiments)} experiments: {experiments}")
    print("=" * 60)
    
    for experiment_name in experiments:
        if experiment_name not in EXPERIMENT_CONFIGS:
            print(f"⚠️  Warning: Experiment '{experiment_name}' not found in configurations")
            continue
            
        config = EXPERIMENT_CONFIGS[experiment_name]
        run_experiment(experiment_name, config, override_params)
        print()

def list_experiments() -> None:
    """List all available experiments."""
    print("Available experiments:")
    print("-" * 40)
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"• {name}: {config['description']}")
        print(f"  Script: {config['script']}")
        print(f"  Parameters: {config['parameters']}")
        print()

def main():
    parser = argparse.ArgumentParser(description="MOSIC Experiment Runner")
    parser.add_argument("--experiments", "-e", nargs="+", 
                       help="List of experiments to run")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available experiments")
    parser.add_argument("--gamma", type=float, help="Override GAMMA parameter")
    parser.add_argument("--alpha", type=float, help="Override ALPHA parameter")
    parser.add_argument("--beta-list", nargs="+", type=float, help="Override beta_list (list of floats)")
    parser.add_argument("--hidden-size-list", nargs="+", type=int, help="Override hidden_size_list (list of ints)")
    parser.add_argument("--l1-lambda", type=float, help="Override L1_LAMBDA parameter")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seeds list")
    parser.add_argument("--expect-group-sizes", nargs="+", type=float, help="Override expect_group_sizes (list of floats)")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # Build override parameters
    override_params = {}
    if args.gamma is not None:
        override_params["GAMMA"] = args.gamma
    if args.alpha is not None:
        override_params["ALPHA"] = args.alpha
    if args.beta_list is not None:
        override_params["beta_list"] = args.beta_list
    if args.hidden_size_list is not None:
        override_params["hidden_size_list"] = args.hidden_size_list
    if args.l1_lambda is not None:
        override_params["L1_LAMBDA"] = args.l1_lambda
    if args.seeds is not None:
        override_params["seeds"] = args.seeds
    if args.expect_group_sizes is not None:
        override_params["expect_group_sizes"] = args.expect_group_sizes
    if args.device is not None:
        override_params["device"] = args.device
    
    run_all_experiments(args.experiments, override_params)

if __name__ == "__main__":
    main() 
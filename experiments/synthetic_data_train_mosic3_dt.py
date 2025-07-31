import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pandas as pd
import sys
sys.path.append('./src')
from sklearn.model_selection import KFold

import pickle
import numpy as np
import torch
from GradTree import GradTreeBlock
# from ps import LinearModel
from MOSIC3 import MOSIC
from eval_utils import evaluate_result_ContBinary, evaluate_result_ContBinary_DR, evaluate_covariate_balance
import random
import ast
import itertools

def get_param_from_env(param_name, default_value, param_type=str):
    """Get parameter from environment variable with fallback to default."""
    env_var = f"PARAM_{param_name}"
    value = os.environ.get(env_var, default_value)
    
    if param_type == list:
        # Handle list parameters (like hidden_size_list)
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                return default_value
        return value
    elif param_type == int:
        return int(value)
    elif param_type == float:
        return float(value)
    else:
        return value

# Data parameters
GAMMA = get_param_from_env("GAMMA", 5, int)

# Model parameters
LR = get_param_from_env("LR", 0.01, float)
L1_LAMBDA = get_param_from_env("L1_LAMBDA", 0.01, float)
identifier_type = get_param_from_env("identifier_type", "dt", str)
assert identifier_type == "dt", "Only DT identifier is supported for this script"

depth_list = get_param_from_env("depth_list", [3], list)
beta_list = get_param_from_env("beta_list", [1e-2, 1e-3], list)

result_dir = get_param_from_env("result_dir", "results", str)

# Get seeds from environment or use default
seeds = get_param_from_env("seeds", [1], list)
if isinstance(seeds, str):
    try:
        seeds = ast.literal_eval(seeds)
    except:
        seeds = [1]

# Get expect_group_sizes from environment
expect_group_sizes = get_param_from_env("expect_group_sizes", [0.5], list)
alphas = get_param_from_env("alphas", [0.0], list)

# Get device from environment
device = get_param_from_env("device", "cuda", str)

for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"seed:{seed}")

    # load data
    with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}.pkl', 'rb') as f:
        train_a, train_y, train_Y1, train_Y0, train_X, train_X_raw, \
        test_a, test_y, test_Y1, test_Y0, test_X, test_X_raw = pickle.load(f)

    with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}-ipw.pkl', 'rb') as f:
        ps_train, ipw_train, ps_test, ipw_test = pickle.load(f)

    with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}-dragonnet-output.pkl', 'rb') as f:
        _, pred_Y1_train, pred_Y0_train, pred_Y1_test, pred_Y0_test = pickle.load(f)

    results = []
    for expect_group_size, alpha in itertools.product(expect_group_sizes, alphas):
        print(f"Expect Group Size:{expect_group_size}, alpha:{alpha}\n")
        
        # Grid search over beta and depth
        best_score = -np.inf
        best_beta = None
        best_depth = None
        best_params = {}
        
        for beta in beta_list:
            for depth in depth_list:
                print(f"Testing beta={beta}, depth={depth}")
                kf = KFold(n_splits=5, random_state=None, shuffle=False)
                val_ef = []
                
                for i, (tr_index, val_index) in enumerate(kf.split(train_X)):
                    print(f"Fold{i}", end='\r')
                    tr_X, val_X = train_X[tr_index], train_X[val_index]
                    tr_a, val_a = train_a[tr_index], train_a[val_index]
                    tr_y, val_y = train_y[tr_index], train_y[val_index]
                    tr_ps, val_ps = ps_train[tr_index], ps_train[val_index]
                    tr_ipw, val_ipw = ipw_train[tr_index], ipw_train[val_index]
                    tr_pred_Y0, val_pred_Y0 = pred_Y0_train[tr_index], pred_Y0_train[val_index]
                    tr_pred_Y1, val_pred_Y1 = pred_Y1_train[tr_index], pred_Y1_train[val_index]

                    # Create identifier model
                    identifier = GradTreeBlock(depth = depth, 
                                               n_estimators = 1, 
                                               n_features = tr_X.shape[1], 
                                               objective = "binary", 
                                               random_seed=0, 
                                               device=device)
                    
                    # Create MOSIC model with current beta
                    model = MOSIC(
                        identifier=identifier,
                        identifier_lr=LR,
                        lambda_lr=LR,
                        beta=beta,
                        l1=L1_LAMBDA,
                        expect_group_size=expect_group_size,
                        alpha=alpha,
                        verbose=False,
                        device=device
                    )
                    
                    model.fit(tr_X, tr_a, tr_y, tr_pred_Y0, tr_pred_Y1, tr_ps, epochs=500)

                    # Get predictions
                    pred_tr = model.predict(torch.tensor(tr_X.astype(np.float32),device=device)).cpu().numpy().flatten()
                    pred_val = model.predict(torch.tensor(val_X.astype(np.float32),device=device)).cpu().numpy().flatten()
                    
                    # Performance on validation set using threshold 0.5
                    sel_idx = (pred_val > 0.5)
                    ate_aiptw = evaluate_result_ContBinary_DR(val_a[sel_idx], val_y[sel_idx], val_pred_Y1[sel_idx], val_pred_Y0[sel_idx], val_ipw[sel_idx])
                    g_size = (sel_idx).mean()
                    val_ef.append(ate_aiptw)

                    # If any fold collapse, skip other folds of that hyper param
                    if np.isnan(ate_aiptw):
                        print(f"Fold{i} collapse")
                        break
                
                # Calculate mean CV score for this combination
                mean_cv_score = np.array(val_ef).mean()
                print(f"  CV Score: {mean_cv_score:.6f}")
                
                # Update best parameters if this combination is better
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_beta = beta
                    best_depth = depth
                    best_params = {
                        'beta': beta,
                        'depth': depth,
                        'cv_score': mean_cv_score
                    }
        
        print(f"Best parameters: beta={best_beta}, depth={best_depth}, cv_score={best_score:.6f}")
        
        # Store best parameters for this expect_group_size
        best_params_info = {
            'beta': best_beta,
            'depth': best_depth,
            'cv_score': best_score
        }
        
        # Create identifier model with best depth
        identifier = GradTreeBlock(depth = 3 if best_depth is None else best_depth, 
                                   n_estimators = 1, 
                                   n_features = train_X.shape[1], 
                                   objective = "binary", 
                                   random_seed=0, 
                                   device=device)
        
        
        # Create MOSIC3 model with best beta
        model_retrain = MOSIC(
            identifier=identifier,
            identifier_lr=LR,
            lambda_lr=LR,
            beta=1e-3 if best_beta is None else best_beta,
            l1=L1_LAMBDA,
            expect_group_size=expect_group_size,
            alpha=alpha,
            verbose=False,
            logger=None,
            device=device
        )
        
        model_retrain.fit(train_X, train_a, train_y, pred_Y0_train, pred_Y1_train, ps_train, epochs=1000,
                          val_X=test_X, val_A=test_a, val_Y=test_y, val_pred_Y0=pred_Y0_test, val_pred_Y1=pred_Y1_test, val_PS=ps_test,
                          constraint_a=None, constraint_coeffs=None, val_constraint_a=None, val_constraint_coeffs=None)
        
        # Get predictions
        pred_train = model_retrain.predict(torch.tensor(train_X.astype(np.float32))).cpu().numpy().flatten()
        pred_test = model_retrain.predict(torch.tensor(test_X.astype(np.float32))).cpu().numpy().flatten()
    
        # tmp_train_size = []
        # tmp_items = []
        # for pred_threshold in np.unique(pred_train): 
        #     sel_idx = (pred_test > pred_threshold)
        #     gt_cate = test_Y1[sel_idx].mean() - test_Y0[sel_idx].mean()
        #     n_unbalance, _, _ = evaluate_covariate_balance(test_X[sel_idx,:], test_a[sel_idx], ipw_test[sel_idx], device=device,smd_threshold = 0.2)
        #     ate_iptw = evaluate_result_ContBinary(test_a[sel_idx], test_y[sel_idx], ipw_test[sel_idx])
        #     ate_aiptw = evaluate_result_ContBinary_DR(test_a[sel_idx], test_y[sel_idx], pred_Y1_test[sel_idx], pred_Y0_test[sel_idx], ipw_test[sel_idx])
        #     ate_cate = (pred_Y1_test[sel_idx])[test_a[sel_idx] == 1].mean() - (pred_Y0_test[sel_idx])[test_a[sel_idx] == 0].mean()
        #     g_size = (sel_idx).mean()
        #     tmp_train_size.append((pred_train > pred_threshold).mean())
        #     tmp_items.append((g_size, gt_cate, ate_iptw, ate_cate, ate_aiptw, n_unbalance, pred_threshold))
    
        # tmp_train_size = np.array(tmp_train_size)
        # tmp_items = np.array(tmp_items)
        # s_idx = np.argmin(np.abs(tmp_train_size - expect_group_size))
        # items.append(tmp_items[s_idx, :])
        # print(tmp_items[s_idx, :])

        # test set
        pred_threshold = 0.5
        test_selected_idx = (pred_test > pred_threshold)
        test_gt_cate = test_Y1[test_selected_idx].mean() - test_Y0[test_selected_idx].mean()
        test_n_unbalance, _, _ = evaluate_covariate_balance(test_X[test_selected_idx,:], test_a[test_selected_idx], ipw_test[test_selected_idx], device=device,smd_threshold = 0.2)
        test_ate_iptw = evaluate_result_ContBinary(test_a[test_selected_idx], test_y[test_selected_idx], ipw_test[test_selected_idx])
        test_ate_aiptw = evaluate_result_ContBinary_DR(test_a[test_selected_idx], test_y[test_selected_idx], pred_Y1_test[test_selected_idx], pred_Y0_test[test_selected_idx], ipw_test[test_selected_idx])
        test_ate_cate = (pred_Y1_test[test_selected_idx])[test_a[test_selected_idx] == 1].mean() - (pred_Y0_test[test_selected_idx])[test_a[test_selected_idx] == 0].mean()
        test_g_size = (test_selected_idx).mean()
        print(f"Test set: group size: {test_g_size}, gt cate: {test_gt_cate}, ate aiptw: {test_ate_aiptw}, n unbalance: {test_n_unbalance}")
        
        # for i1, i2, i3, i4, i5, i6, i7 in items:
        #     print(i1, i2, i3, i4, i5, i6)

        result = {
            'results': {
                'test_g_size': test_g_size,
                'test_gt_cate': test_gt_cate,
                'test_ate_aiptw': test_ate_aiptw,
                'test_n_unbalance': test_n_unbalance,        
            },
            'constraint_params': {
                'expect_group_size': expect_group_size,
                'alpha': alpha,
            },
            'best_parameters': best_params_info,
            'grid_search_params': {'beta_list': beta_list, 'depth_list': depth_list}
        }
        results.append(result)
    
    # Save results with grid search info
    filename = f"mosic3_{identifier_type}_gridsearch_seed{seed}.pkl"
    with open(os.path.join(result_dir, f"GAMMA_{GAMMA}", filename), 'wb') as f:
        pickle.dump(results, f)
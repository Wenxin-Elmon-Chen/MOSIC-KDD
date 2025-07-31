from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import torch
from utils import standardized_mean_difference
from ps import PropensityEstimator, cal_weights_align
from DecisionTree import DecisionTree


def evaluate_result_ContBinary(A, Y, weight):
    avg1 = np.sum(Y * A * weight) / np.sum(A * weight)
    avg0 = np.sum(Y * (1 - A) * weight) / np.sum((1 - A) * weight)
    return avg1 - avg0


def evaluate_result_ContBinary_DR(A, Y, pred_Y1, pred_Y0, weight):
    item1 = np.sum((pred_Y1 - pred_Y0))
    item2 = np.sum(weight * A * (Y - pred_Y1))
    item3 = np.sum(weight * (1 - A) * (Y - pred_Y0))
    return (item1 + item2 - item3) / len(Y)

def evaluate_covariate_balance(X, A, weight, device, smd_threshold=0.2):
    if not torch.is_tensor(X):
        X = torch.tensor(X).to(device).float()
    if not torch.is_tensor(weight):
        weight = torch.tensor(weight).to(device).float()
    if not torch.is_tensor(A):
        A = torch.tensor(A).to(device)
    a1_mask = (A.view(-1, 1) == 1).float()
    a0_mask = (A.view(-1, 1) == 0).float()
    smd = standardized_mean_difference(a1_mask, a0_mask, weight,X)
    n_unbalanced_feature_weighted = torch.sum(smd > smd_threshold).item()
    max_smd_weighted = torch.max(smd).item()

    return n_unbalanced_feature_weighted, max_smd_weighted, smd

def calculate_odds_ratio(A, Y, W):
    
    A = np.asarray(A)
    Y = np.asarray(Y)
    W = np.asarray(W)
    
    # Treated group (A == 1)
    treated_mask = A == 1
    Y1_treated = np.sum(W[treated_mask & (Y != 0)])  # Weighted Y=1 in treated
    Y0_treated = np.sum(W[treated_mask & (Y == 0)])  # Weighted Y=0 in treated
    
    # Control group (A == 0)
    control_mask = A == 0
    Y1_control = np.sum(W[control_mask & (Y != 0)])  # Weighted Y=1 in control
    Y0_control = np.sum(W[control_mask & (Y == 0)])  # Weighted Y=0 in control
    
    # Compute odds for treated and control groups
    odds_treated = Y1_treated / Y0_treated if Y0_treated > 0 else np.inf
    odds_control = Y1_control / Y0_control if Y0_control > 0 else np.inf
    
    # Calculate Odds Ratio
    odds_ratio = odds_treated / odds_control if odds_control > 0 else np.inf
    
    return odds_ratio


def evaluate_model_softprob_ps(x, a, y, subgroup_weight, ps, ps_truncation=False, smd_threshold=0.2, train_ps_on_test = False):
    # Is Y survival outcome or continuous outcome?
    if len(y.shape) == 2:
        y_type = "survival"
    elif len(y.shape) == 1:
        # Determine if Y is binary or continuous
        if len(np.unique(y)) == 2:
            y_type = "binary"
        else:
            y_type = "continuous"
    else:
        raise ValueError("Y should be 1D or 2D array")

    # Convert to tensor
    a_tensor = torch.tensor(a)
    x_tensor = torch.tensor(x)
    A_1_mask = (a_tensor.view(-1, 1) == 1).float()
    A_0_mask = (a_tensor.view(-1, 1) == 0).float()

    ### Use propensity model trained on train set ###
    # Get IPTW
    # pre_ps_train = ps_model.predict_proba(x)[:, 1]
    pre_ps_train = ps
    _, _, ipw_ps_train = cal_weights_align(a, pre_ps_train, normalized=True, stabilized=False, clip=ps_truncation)
    ipw_ps_train = ipw_ps_train[:, 0]
    
    ipw_ps_train_tensor = torch.tensor(ipw_ps_train)

    # Evaluate Balance
    smd_ps_train = standardized_mean_difference(A_1_mask,A_0_mask, ipw_ps_train_tensor * subgroup_weight.flatten(), x_tensor)
    num_unbalance_ps_train = (smd_ps_train > smd_threshold).sum().item()

    # Evaluate Group size
    group_size = np.mean(subgroup_weight).item()

    # Evaluate subgroup ATE
    if y_type == "survival":
        # Hazard ratio
        try:
            cate_ps_train, cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = \
                                                                evaluate_result_HR(x, a, y, ipw_ps_train * subgroup_weight.flatten())
        except Exception as e:
            cate_ps_train, cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan, np.nan
            print(f"Error: {e}")
    elif y_type == "continuous":
        # Continous subgroup ATE
        cate_ps_train = evaluate_result_ContBinary(a, y, ipw_ps_train * subgroup_weight.flatten())
        cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan # TODO: implement CI for Continous CATE
    elif y_type == "binary":
        # Binary subgroup ATE (Odds Ratio)
        cate_ps_train = calculate_odds_ratio(a, y, ipw_ps_train * subgroup_weight.flatten())
        cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan # TODO: implement CI for Binary ATE
    else:
        raise ValueError("Unknown Y type")

    ### Refit propensity model on test set ###
    if train_ps_on_test:
        ps = PropensityEstimator(learner="LR", 
                                    paras_grid = {'C': 10 ** np.arange(-3, 3, 0.5)}, 
                                    criteria="balance",
                                    print_result=False)
        ps.cross_validation_fit(x, a)
        pre = ps.best_model.predict_proba(x)[:, 1]
        _, _, ipw = cal_weights_align(a, pre, normalized=True,stabilized=False, clip=ps_truncation)
        ipw = ipw[:, 0]
        ipw_tensor = torch.tensor(ipw)

        # Evaluate Balance
        smd = standardized_mean_difference(A_1_mask, A_0_mask, ipw_tensor * subgroup_weight.flatten(), x_tensor)
        num_unbalance = (smd > smd_threshold).sum().item()

        if y_type == "survival":
            # Hazard ratio
            try:
                cate, cate_se, cate_ci_lower, cate_ci_upper = evaluate_result_HR(x, a, y, ipw * subgroup_weight.flatten())
            except Exception as e:
                cate, cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan, np.nan
                print(f"Error: {e}")
        elif y_type == "continuous":
            # Continuous subgroup ATE
            cate = evaluate_result_ContBinary(a, y, ipw * subgroup_weight.flatten())
            cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan # TODO: implement CI for Continous ATE
        elif y_type == "binary":
            # Binary subgroup ATE (Odds Ratio)
            cate = calculate_odds_ratio(a, y, ipw * subgroup_weight.flatten())
            cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan # TODO: implement CI for Binary ATE
    else:
        num_unbalance = np.nan
        smd = np.nan
        cate = np.nan
        cate_se = np.nan
        cate_ci_lower = np.nan
        cate_ci_upper = np.nan

    return {"num_unbalance":num_unbalance, "num_unbalance_ps_train":num_unbalance_ps_train,
            "cate":cate,"cate_ps_train":cate_ps_train,
            "cate_se":cate_se,"cate_se_ps_train":cate_se_ps_train,
            "cate_ci_lower":cate_ci_lower,"cate_ci_upper":cate_ci_upper,
            "cate_ci_lower_ps_train":cate_ci_lower_ps_train,"cate_ci_upper_ps_train":cate_ci_upper_ps_train,
            "group_size":group_size,
            "smd":smd,"smd_ps_train":smd_ps_train}



def evaluate_model_softprob(x, a, y, subgroup_weight, ps_model, ps_truncation=False, smd_threshold=0.2, train_ps_on_test = False):
    # Is Y survival outcome or continuous outcome?
    if len(y.shape) == 2:
        y_type = "survival"
    elif len(y.shape) == 1:
        # Determine if Y is binary or continuous
        if len(np.unique(y)) == 2:
            y_type = "binary"
        else:
            y_type = "continuous"
    else:
        raise ValueError("Y should be 1D or 2D array")

    # Convert to tensor
    a_tensor = torch.tensor(a)
    x_tensor = torch.tensor(x)
    A_1_mask = (a_tensor.view(-1, 1) == 1).float()
    A_0_mask = (a_tensor.view(-1, 1) == 0).float()

    ### Use propensity model trained on train set ###
    # Get IPTW
    pre_ps_train = ps_model.predict_proba(x)[:, 1]
    _, _, ipw_ps_train = cal_weights_align(a, pre_ps_train, normalized=True, stabilized=False, clip=ps_truncation)
    ipw_ps_train = ipw_ps_train[:, 0]
    
    ipw_ps_train_tensor = torch.tensor(ipw_ps_train)

    # Evaluate Balance
    smd_ps_train = standardized_mean_difference(A_1_mask,A_0_mask, ipw_ps_train_tensor * subgroup_weight.flatten(), x_tensor)
    num_unbalance_ps_train = (smd_ps_train > smd_threshold).sum().item()

    # Evaluate Group size
    group_size = np.mean(subgroup_weight).item()

    # Evaluate subgroup ATE
    if y_type == "survival":
        # Hazard ratio
        try:
            cate_ps_train, cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = \
                                                                evaluate_result_HR(x, a, y, ipw_ps_train * subgroup_weight.flatten())
        except Exception as e:
            cate_ps_train, cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan, np.nan
            print(f"Error: {e}")
    elif y_type == "continuous":
        # Continous subgroup ATE
        cate_ps_train = evaluate_result_ContBinary(a, y, ipw_ps_train * subgroup_weight.flatten())
        cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan # TODO: implement CI for Continous CATE
    elif y_type == "binary":
        # Binary subgroup ATE (Odds Ratio)
        cate_ps_train = calculate_odds_ratio(a, y, ipw_ps_train * subgroup_weight.flatten())
        cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan # TODO: implement CI for Binary ATE
    else:
        raise ValueError("Unknown Y type")

    ### Refit propensity model on test set ###
    if train_ps_on_test:
        ps = PropensityEstimator(learner="LR", 
                                    paras_grid = {'C': 10 ** np.arange(-3, 3, 0.5)}, 
                                    criteria="balance",
                                    print_result=False)
        ps.cross_validation_fit(x, a)
        pre = ps.best_model.predict_proba(x)[:, 1]
        _, _, ipw = cal_weights_align(a, pre, normalized=True,stabilized=False, clip=ps_truncation)
        ipw = ipw[:, 0]
        ipw_tensor = torch.tensor(ipw)

        # Evaluate Balance
        smd = standardized_mean_difference(A_1_mask, A_0_mask, ipw_tensor * subgroup_weight.flatten(), x_tensor)
        num_unbalance = (smd > smd_threshold).sum().item()

        if y_type == "survival":
            # Hazard ratio
            try:
                cate, cate_se, cate_ci_lower, cate_ci_upper = evaluate_result_HR(x, a, y, ipw * subgroup_weight.flatten())
            except Exception as e:
                cate, cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan, np.nan
                print(f"Error: {e}")
        elif y_type == "continuous":
            # Continuous subgroup ATE
            cate = evaluate_result_ContBinary(a, y, ipw * subgroup_weight.flatten())
            cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan # TODO: implement CI for Continous ATE
        elif y_type == "binary":
            # Binary subgroup ATE (Odds Ratio)
            cate = calculate_odds_ratio(a, y, ipw * subgroup_weight.flatten())
            cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan # TODO: implement CI for Binary ATE
    else:
        num_unbalance = np.nan
        smd = np.nan
        cate = np.nan
        cate_se = np.nan
        cate_ci_lower = np.nan
        cate_ci_upper = np.nan

    return {"num_unbalance":num_unbalance, "num_unbalance_ps_train":num_unbalance_ps_train,
            "cate":cate,"cate_ps_train":cate_ps_train,
            "cate_se":cate_se,"cate_se_ps_train":cate_se_ps_train,
            "cate_ci_lower":cate_ci_lower,"cate_ci_upper":cate_ci_upper,
            "cate_ci_lower_ps_train":cate_ci_lower_ps_train,"cate_ci_upper_ps_train":cate_ci_upper_ps_train,
            "group_size":group_size,
            "smd":smd,"smd_ps_train":smd_ps_train}

def evaluate_model_hard(x, a, y, subgroup_mask, ps_model, ps_truncation=False, smd_threshold=0.2, train_ps_on_test = False):
    
    # Is Y survival outcome or continuous outcome?
    if len(y.shape) == 2:
        y_type = "survival"
    elif len(y.shape) == 1:
        # Determine if Y is binary or continuous
        if len(np.unique(y)) == 2:
            y_type = "binary"
        else:
            y_type = "continuous"
    else:
        raise ValueError("Y should be 1D or 2D array")

    subgroup_mask = subgroup_mask.flatten()
    # Convert to tensor
    a_tensor = torch.tensor(a)    
    x_tensor = torch.tensor(x)
    A_1_mask = (a_tensor.view(-1, 1) == 1).float()
    A_0_mask = (a_tensor.view(-1, 1) == 0).float()
    A_1_mask_subgroup = A_1_mask[subgroup_mask]
    A_0_mask_subgroup = A_0_mask[subgroup_mask]
    x_tensor_subgroup = x_tensor[subgroup_mask, :]

    ### Use propensity model trained on train set ###
    # Get IPTW
    pre_ps_train = ps_model.predict_proba(x)[:, 1]
    _, _, ipw_ps_train = cal_weights_align(a, pre_ps_train, normalized=True, stabilized=False, clip=ps_truncation)
    ipw_ps_train = ipw_ps_train[:, 0]

    ipw_ps_train_tensor = torch.tensor(ipw_ps_train)
    ipw_ps_train_tensor_subgroup = ipw_ps_train_tensor[subgroup_mask]
    
    # Evaluate Balance
    smd_ps_train = standardized_mean_difference(A_1_mask_subgroup,A_0_mask_subgroup,
                                                ipw_ps_train_tensor_subgroup,x_tensor_subgroup)
    num_unbalance_ps_train = (smd_ps_train > smd_threshold).sum().item()

    # Group size
    group_size = subgroup_mask.sum()/len(subgroup_mask)

    # Evaluate subgroup ATE
    if y_type == "survival":
        # Hazard ratio
        try:
            cate_ps_train, cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = \
                                                                evaluate_result_HR(x[subgroup_mask,:], 
                                                                    a[subgroup_mask], 
                                                                    y[subgroup_mask,:], 
                                                                    ipw_ps_train[subgroup_mask],
                                                                    interaction=False)
        except Exception as e:
            cate_ps_train, cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan, np.nan
            print(f"Error: {e}")
    elif y_type == "continuous":
        # Continuous subgroup ATE
        cate_ps_train = evaluate_result_ContBinary(a[subgroup_mask], y[subgroup_mask], ipw_ps_train[subgroup_mask])
        cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan # TODO: implement CI for Continous CATE
    elif y_type == "binary":
        # Binary subgroup ATE (Odds Ratio)
        cate_ps_train = calculate_odds_ratio(a[subgroup_mask], y[subgroup_mask], ipw_ps_train[subgroup_mask])
        cate_se_ps_train, cate_ci_lower_ps_train, cate_ci_upper_ps_train = np.nan, np.nan, np.nan # TODO: implement CI for Binary CATE
    else:
        raise ValueError("Unknown Y type")
        

    ### Refit propensity model on test set ###
    if train_ps_on_test:
        ps = PropensityEstimator(learner="LR", 
                                    paras_grid = {'C': 10 ** np.arange(-3, 3, 0.5)}, 
                                    criteria="balance",
                                    print_result=False)
        ps.cross_validation_fit(x, a)
        pre = ps.best_model.predict_proba(x)[:, 1]
        _, _, ipw = cal_weights_align(a, pre, normalized=True,stabilized=False, clip=ps_truncation)
        ipw = ipw[:, 0]

        ipw_tensor = torch.tensor(ipw)
        ipw_tensor_subgroup = ipw_tensor[subgroup_mask]
        
        # Evaluate Balance
        smd = standardized_mean_difference(A_1_mask_subgroup, A_0_mask_subgroup, 
                                            ipw_tensor_subgroup, x_tensor_subgroup)
        num_unbalance = (smd > smd_threshold).sum().item()

        # Evaluate subgroup ATE
        if y_type == "survival":
            # Hazard ratio
            try:
                cate, cate_se, cate_ci_lower, cate_ci_upper = evaluate_result_HR(x[subgroup_mask,:], 
                                                    a[subgroup_mask], 
                                                    y[subgroup_mask,:], 
                                                    ipw[subgroup_mask])
            except Exception as e:
                cate, cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan, np.nan
                print(f"Error: {e}")
            
        elif y_type == "continuous":
            # Continuous subgroup ATE
            cate = evaluate_result_ContBinary(a[subgroup_mask], y[subgroup_mask], ipw[subgroup_mask])
            cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan # TODO: implement CI for Continous CATE
        elif y_type == "binary":
            # Binary subgroup ATE (Odds Ratio)
            cate = calculate_odds_ratio(a[subgroup_mask], y[subgroup_mask], ipw[subgroup_mask])
            cate_se, cate_ci_lower, cate_ci_upper = np.nan, np.nan, np.nan # TODO: implement CI for Binary CATE
        else:
            raise ValueError("Unknown Y type")
    else:
        num_unbalance = np.nan
        smd = np.nan
        cate = np.nan
        cate_se = np.nan
        cate_ci_lower = np.nan
        cate_ci_upper = np.nan

    return {"num_unbalance":num_unbalance, "num_unbalance_ps_train":num_unbalance_ps_train,
            "cate":cate,"cate_ps_train":cate_ps_train,
            "cate_se":cate_se,"cate_se_ps_train":cate_se_ps_train,
            "cate_ci_lower":cate_ci_lower,"cate_ci_upper":cate_ci_upper,
            "cate_ci_lower_ps_train":cate_ci_lower_ps_train,"cate_ci_upper_ps_train":cate_ci_upper_ps_train,
            "group_size":group_size,
            "smd":smd,"smd_ps_train":smd_ps_train}

def plotGradTree(tree_split_values, tree_split_index, leaf_classes_array, train_X, X_col_names, scales, means):

    tree = DecisionTree(tree_split_values[0,:,:], tree_split_index[0,:,:], leaf_classes_array[0,:])
    tree.prune_tree(data=train_X, min_samples=1)

    tree.plot_tree(feature_names=X_col_names,scales=scales,means=means)
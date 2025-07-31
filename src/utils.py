import pandas as pd
import numpy as np
import torch

def fill_na_value(data, X_cols):
    for _ in X_cols:
        data[_] = data[_].fillna(data[_].median())
    data = data.dropna(axis=1, how='all')
    assert not data.isnull().values.any()
    return data

def phi(event_times, event_observed):
    """
    Calculate ranking loss for a given set of event times and event indicators.
    phi((y_i, t_i),(y_j, t_j)) = 1 if t_i < t_j and y_i ==1 ; -1 if t_i > t_j and y_j == 1; 0 otherwise

    Parameters:
    # event_times: np.ndarray of event/censoring times. (n,)
    # event_observed: np.ndarray of event indicators (1 if the event was observed, 0 if censored). (n,)
    # event_times: torch.tensor of event/censoring times. (n,)
    # event_observed: torch.tensor of event indicators (1 if the event was observed, 0 if censored). (n,)

    Returns:
    torch.tensor of pairwise ranking loss. (n, n)
    """
    n = event_times.shape[0]

    # Create pairwise comparisons
    event_times_i = event_times.unsqueeze(1).expand(n, n).T
    event_times_j = event_times_i.T

    event_observed_i = event_observed.unsqueeze(1).expand(n, n).T
    event_observed_j = event_observed_i.T

    # Calculate the ranking loss
    phi_mat = torch.zeros((n, n), device=event_times.device)
    i_gt_j_mask = ((event_times_i > event_times_j) & (event_observed_j == 1)).T
    i_lt_j_mask = ((event_times_i < event_times_j) & (event_observed_i == 1)).T
    phi_mat[i_gt_j_mask] = 1
    phi_mat[i_lt_j_mask] = -1

    return phi_mat

def mean_difference(treatment_mask, control_mask, weights, X):
    # all the inputs are torch tensors
    # treatment_mask: torch.Tensor, shape (n, 1)
    # control_mask: torch.Tensor, shape (n, 1)
    # weights: torch.Tensor, shape (n, 1)
    # y: torch.Tensor, shape (n, 1)
    # X: torch.Tensor, shape (n, p)
    ww_1 = treatment_mask * weights.view(-1, 1)
    ww_0 = control_mask * weights.view(-1, 1)
    mu_1 = (X.T @ ww_1) / torch.sum(ww_1)
    mu_0 = (X.T @ ww_0) / torch.sum(ww_0)
    return torch.abs(mu_1 - mu_0)


def standardized_mean_difference(treatment_mask, control_mask, weights, X):
    # all the inputs are torch tensors
    # treatment_mask: torch.Tensor, shape (n, 1)
    # control_mask: torch.Tensor, shape (n, 1)
    # weights: torch.Tensor, shape (n, 1)
    # y: torch.Tensor, shape (n, 1)
    # X: torch.Tensor, shape (n, p)
    ww_1 = treatment_mask * weights.view(-1, 1)
    ww_0 = control_mask * weights.view(-1, 1)
    mu_1 = (X.T @ ww_1) / torch.sum(ww_1)
    mu_0 = (X.T @ ww_0) / torch.sum(ww_0)

    s_1_square = torch.square(X.T - mu_1) @ ww_1 / (
            torch.sum(ww_1) - torch.sum(torch.square(ww_1)) / torch.sum(ww_1))
    s_0_square = torch.square(X.T - mu_0) @ ww_0 / (
            torch.sum(ww_0) - torch.sum(torch.square(ww_0)) / torch.sum(ww_0))
    # smd = torch.relu(torch.abs(mu_1 - mu_0) / (torch.sqrt((s_1_square + s_0_square) / 2)) - 0.1)
    smd = torch.abs(mu_1 - mu_0) / (torch.sqrt((s_1_square + s_0_square) / 2))
    return smd

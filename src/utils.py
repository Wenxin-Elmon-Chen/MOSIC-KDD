import pandas as pd
import numpy as np
import torch

def fill_na_value(data, X_cols):
    for _ in X_cols:
        data[_] = data[_].fillna(data[_].median())
    data = data.dropna(axis=1, how='all')
    assert not data.isnull().values.any()
    return data

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

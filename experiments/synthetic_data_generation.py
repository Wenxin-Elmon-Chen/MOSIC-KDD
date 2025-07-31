import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import sys
sys.path.append('./src')
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
# import tensorflow as tf
import random
import torch

N = 5000
P = 10
P_STAR = 4
OMEGA = 2


GAMMA_INTERCEPT = 0
RHO = 0.3
BETA_0_TILDE = 0
BETA_1_TILDE = 2
BETA_TAU_TILDE = 0.5
N_SEED = 100


def generate_multivariate_normal(p, n, sigma_X, rho):
    # Create identity matrix and ones matrix
    I_p = np.eye(p)  # Identity matrix of size p
    ones_p = np.ones((p, p))  # Matrix of ones
    
    # Covariance matrix: sigma_X^2 * [(1 - rho) * I_p + rho * 1_p * 1_p^T]
    cov_matrix = sigma_X**2 * ((1 - rho) * I_p + rho * ones_p)
    
    # Mean vector (assuming zero mean for simplicity)
    mean_vector = np.zeros(p)
    
    # Draw n samples from the multivariate normal distribution
    samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=n)
    
    return samples

def generate_data_continuous(supp_beta_0, supp_beta_1, supp_beta_tau, supp_gamma, gamma_intercept,
                             N, p, rho, sigma_X, sigma_y, gamma_tilde, beta_0_tilde, beta_1_tilde, beta_tau_tilde, theta, add_term = "sq", verbose=False):
    '''
    Inputs:
    supp_beta_0: support of beta_0
    supp_beta_tau: support of beta_tau
    supp_gamma: support of gamma
    N: number of samples
    p: dimension of covariates
    rho: correlation between covariates
    sigma_X: standard deviation of covariates
    sigma_y: standard deviation of noise
    a: shape of weibull distribution. When a=1, it collapsed to exponential distribution
    b_censor: scale of weibull distribution for censoring time
    gamma_tilde: imbalance parameter
    beta_0_tilde: effective magnitude of beta_0
    beta_tau_tilde: effective magnitude of beta_tau
    theta: ATE
    '''
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # indicator function for beta and gamma, where `*_tilde` position has 1, and others have 0
    I_G, I_beta_0, I_beta_1, I_beta_tau = np.zeros(p), np.zeros(p), np.zeros(p), np.zeros(p)
    I_G[supp_gamma] = 1
    I_beta_0[supp_beta_0] = 1
    I_beta_1[supp_beta_1] = 1
    I_beta_tau[supp_beta_tau] = 1
    gamma = gamma_tilde * I_G
    beta_0 = beta_0_tilde * I_beta_0
    beta_1 = beta_1_tilde * I_beta_1
    beta_tau = beta_tau_tilde * I_beta_tau
    # Randomly assign positive/negative sign to beta_0, beta_tau, and gamma
    gamma = gamma * np.random.choice([-1, 1], p)
    gamma[supp_gamma[-1]] = 2 * gamma[supp_gamma[-1]]
    for idx in [-1]:
        if gamma[supp_gamma[idx]] > 0:
            gamma[supp_gamma[idx]] = - gamma[supp_gamma[idx]]

    # Generate data
    X = generate_multivariate_normal(p, N, sigma_X, rho)
    T = np.random.binomial(1,_sigmoid(X @ gamma + gamma_intercept))
    epsilon = np.random.normal(0,sigma_y**2,N)

    if add_term == "sq":
        Y0 = X @ beta_0 + (np.sin(10 * X) + 5 * (X ** 2)) @ beta_1 + epsilon
        Y1 = X @ beta_0 + (np.sin(10 * X) + 5 * (X ** 2)) @ beta_1 + X @ beta_tau + theta + epsilon
    else:
        Y0 = X @ beta_0 + epsilon
        Y1 = X @ beta_0 + X @ beta_tau + theta + epsilon
    Y = T * Y1 + (1 - T) * Y0

    data = np.column_stack((X, T, Y))
    data = pd.DataFrame(data, columns=[f'X{i}' for i in range(p)] + ['T', 'Y'])
    hte = Y1 - Y0
    if verbose:
        print(f'gamma: {gamma},\n beta_0: {beta_0},\n beta_1: {beta_1}, \n beta_tau: {beta_tau}')
        print(f'ATE: {hte.mean()}')

    return Y0, Y1, gamma, beta_0, beta_tau, data

def read_synthetic_continuous(N, p, p_star, rho, omega,gamma_tilde,gamma_intercept,
                              beta_0_tilde,beta_1_tilde,beta_tau_tilde,theta,seed=42,verbose=False,add_term='sq',sigma_X=0.1,sigma_y=0.1):
    np.random.seed(seed)

    supp_beta_1 = np.linspace(2*p_star - omega - 1, 2*p_star - omega - 1, 1, dtype=int)
    supp_beta_0 = np.linspace(0, p_star-1, p_star, dtype=int) 
    supp_beta_tau = np.linspace(0, int(p_star) - 1, int(p_star), dtype=int) 
    supp_gamma = np.linspace(p_star - omega, 2*p_star - omega - 1, p_star, dtype=int)
    

    y0, y1, _, _, _, df = generate_data_continuous(supp_beta_0, supp_beta_1, supp_beta_tau, supp_gamma,gamma_intercept, 
                                                   N, p, rho, sigma_X, sigma_y, gamma_tilde, beta_0_tilde, beta_1_tilde, beta_tau_tilde, theta, add_term=add_term,verbose = verbose)
    a_data = df['T'].values
    y_data = df['Y'].values
    sel_cols = [x for x in df.columns if 'X' in x]
    X_data = df[sel_cols].values

    return a_data, y_data, X_data, sel_cols, y0, y1


for GAMMA in [0.0, 5.0]:

    for seed in range(1, N_SEED + 1):
        # tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # Generate data
        a_data, y_data, X_data, sel_cols, Y0, Y1 = read_synthetic_continuous(N = N, p = P, p_star = P_STAR, rho = RHO, omega = OMEGA, 
                                                                            gamma_tilde = GAMMA, gamma_intercept=GAMMA_INTERCEPT,
                                                                            beta_0_tilde = BETA_0_TILDE,beta_1_tilde=BETA_1_TILDE,beta_tau_tilde = BETA_TAU_TILDE, theta = 0, seed = seed,
                                                                            add_term='sq')           
        
        train_index, test_index = train_test_split(range(X_data.shape[0]), test_size=0.5, random_state=seed)
        train_a, test_a = a_data[train_index], a_data[test_index]
        train_y, test_y = y_data[train_index], y_data[test_index]
        train_X_raw, test_X_raw = X_data[train_index,:], X_data[test_index,:]
        
        imp = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        train_X = imp.fit_transform(train_X_raw)
        train_X = scaler.fit_transform(train_X)
        test_X = imp.transform(test_X_raw)
        test_X = scaler.transform(test_X)
        
        train_X, test_X = train_X, test_X
        train_Y0, test_Y0 = Y0[train_index], Y0[test_index]
        train_Y1, test_Y1 = Y1[train_index], Y1[test_index]
        
        with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}.pkl', 'wb') as f:
            pickle.dump((train_a, train_y, train_Y1, train_Y0, train_X, train_X_raw, 
                        test_a, test_y, test_Y1, test_Y0, test_X, test_X_raw), f)
        
        with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}-scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
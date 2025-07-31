import sys

# for linux env.
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
from sklearn.metrics import log_loss
from tqdm import tqdm
# import xgboost as xgb
# import lightgbm as lgb
# from iptw.evaluation import cal_deviation, SMD_THRESHOLD, cal_weights, model_eval_common_simple
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
from lifelines import KaplanMeierFitter, CoxPHFitter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SMD_THRESHOLD = 0.2

# %%  Aux-functions
def logits_to_probability(logits, normalized):
    if normalized:
        if len(logits.shape) == 1:
            return logits
        elif len(logits.shape) == 2:
            return logits[:, 1]
        else:
            raise ValueError
    else:
        if len(logits.shape) == 1:
            return 1 / (1 + np.exp(-logits))
        elif len(logits.shape) == 2:
            prop = softmax(logits, axis=1)
            return prop[:, 1]
        else:
            raise ValueError

def weighted_mean(x, w):
    # input: x: n * d, w: n * 1
    # output: d
    x_w = np.multiply(x, w)
    n_w = np.sum(w)  # w.sum()
    m_w = np.sum(x_w, axis=0) / n_w
    return m_w

def weighted_var(x, w):
    # x: n * d, w: n * 1
    m_w = weighted_mean(x, w)  # d
    # nw, nsw = w.sum(), (w ** 2).sum()
    nw, nsw = np.sum(w), np.sum(w ** 2)
    var = np.multiply((x - m_w) ** 2, w)  # n*d
    var = np.sum(var, axis=0) * (nw / (nw ** 2 - nsw))
    return var

def smd_func(x1, w1, x0, w0, abs=True):
    w_mu1, w_var1 = weighted_mean(x1, w1), weighted_var(x1, w1)
    w_mu0, w_var0 = weighted_mean(x0, w0), weighted_var(x0, w0)
    VAR_w = np.sqrt((w_var1 + w_var0) / 2)
    smd_result = np.divide(
        (w_mu1 - w_mu0),
        VAR_w, out=np.zeros_like(w_mu1), where=VAR_w != 0)
    if abs:
        smd_result = np.abs(smd_result)
    return smd_result

def cal_weights_align(golds_treatment, logits_treatment, normalized, stabilized=True, clip=True):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    logits_treatment = logits_to_probability(logits_treatment, normalized)
    p_T = len(ones_idx[0]) / (len(ones_idx[0]) + len(zeros_idx[0]))

    # comment out p_T scaled IPTW
    if stabilized:
        # stabilized weights:   treated_w.sum() + controlled_w.sum() ~ N
        treated_w, controlled_w = p_T / logits_treatment[ones_idx], (1 - p_T) / (
                1. - logits_treatment[zeros_idx])  # why *p_T here?

    else:
        # standard IPTW:  treated_w.sum() + controlled_w.sum() > N
        treated_w, controlled_w = 1. / logits_treatment[ones_idx], 1. / (1. - logits_treatment[zeros_idx])  # why *p_T here? my added test

    treated_w[np.isinf(treated_w)] = 0
    controlled_w[np.isinf(controlled_w)] = 0

    # print(treated_w, controlled_w)

    if clip:
        # treated_w = np.clip(treated_w, a_min=1e-06, a_max=50)
        # controlled_w = np.clip(controlled_w, a_min=1e-06, a_max=50)
        amin = np.quantile(np.concatenate((treated_w, controlled_w)), 0.01)
        amax = np.quantile(np.concatenate((treated_w, controlled_w)), 0.99)

        if amax > 50:
            # if there are inf involved in qunatile, returen nan
            amax = np.quantile(np.concatenate((treated_w, controlled_w)), 0.8)
        if amin <= 1e-6:
            amin = np.quantile(np.concatenate((treated_w, controlled_w)), 0.2)

        # print('Using IPTW trim [{}, {}]'.format(amin, amax))
        treated_w = np.clip(treated_w, a_min=amin, a_max=amax)
        controlled_w = np.clip(controlled_w, a_min=amin, a_max=amax)

    # treated_w = np.where(treated_w < 10, treated_w, 25)
    # controlled_w = np.where(controlled_w < 10, controlled_w, 10)

    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w), 1)), np.reshape(controlled_w,
                                                                                     (len(controlled_w), 1))

    all_w = np.zeros((len(treated_w) + len(controlled_w), 1))
    for arr_idx in range(len(treated_w)):
        all_idx = ones_idx[0][arr_idx]
        all_w[all_idx, 0] = treated_w[arr_idx, 0]

    for arr_idx in range(len(controlled_w)):
        all_idx = zeros_idx[0][arr_idx]
        all_w[all_idx, 0] = controlled_w[arr_idx, 0]

    return treated_w, controlled_w, all_w

def cal_weights(golds_treatment, logits_treatment, normalized, stabilized=True, clip=True):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    logits_treatment = logits_to_probability(logits_treatment, normalized)
    p_T = len(ones_idx[0]) / (len(ones_idx[0]) + len(zeros_idx[0]))

    # comment out p_T scaled IPTW
    if stabilized:
        # stabilized weights:   treated_w.sum() + controlled_w.sum() ~ N
        treated_w, controlled_w = p_T / logits_treatment[ones_idx], (1 - p_T) / (
                1. - logits_treatment[zeros_idx])  # why *p_T here?

    else:
        # standard IPTW:  treated_w.sum() + controlled_w.sum() > N
        treated_w, controlled_w = 1. / logits_treatment[ones_idx], 1. / (
                1. - logits_treatment[zeros_idx])  # why *p_T here? my added test

    treated_w[np.isinf(treated_w)] = 0
    controlled_w[np.isinf(controlled_w)] = 0

    if clip:
        # treated_w = np.clip(treated_w, a_min=1e-06, a_max=50)
        # controlled_w = np.clip(controlled_w, a_min=1e-06, a_max=50)
        amin = np.quantile(np.concatenate((treated_w, controlled_w)), 0.01)
        amax = np.quantile(np.concatenate((treated_w, controlled_w)), 0.99)

        if amax > 50:
            # if there are inf involved in qunatile, returen nan
            amax = np.quantile(np.concatenate((treated_w, controlled_w)), 0.8)
        if amin <= 1e-6:
            amin = np.quantile(np.concatenate((treated_w, controlled_w)), 0.2)

        # print('Using IPTW trim [{}, {}]'.format(amin, amax))
        treated_w = np.clip(treated_w, a_min=amin, a_max=amax)
        controlled_w = np.clip(controlled_w, a_min=amin, a_max=amax)

    # treated_w = np.where(treated_w < 10, treated_w, 25)
    # controlled_w = np.where(controlled_w < 10, controlled_w, 10)

    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w), 1)), np.reshape(controlled_w,
                                                                                     (len(controlled_w), 1))
    return treated_w, controlled_w

def cal_deviation(hidden_val, golds_treatment, logits_treatment, normalized, verbose=1):
    # covariates, and IPTW
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized=normalized)
    if verbose:
        print('In cal_deviation: n_treated:{}, n_treated_w:{} |'
              'n_controlled:{}, n_controlled_w:{} |'
              'n:{}, n_w:{}'.format(len(treated_w), treated_w.sum(), len(controlled_w), controlled_w.sum(),
                                    len(golds_treatment), treated_w.sum() + controlled_w.sum()))
    hidden_val = np.asarray(hidden_val)  # original covariates, to be weighted
    hidden_treated, hidden_controlled = hidden_val[ones_idx], hidden_val[zeros_idx]

    # Original SMD
    hidden_treated_mu, hidden_treated_var = np.mean(hidden_treated, axis=0), np.var(hidden_treated, axis=0, ddof=1)
    hidden_controlled_mu, hidden_controlled_var = np.mean(hidden_controlled, axis=0), np.var(hidden_controlled, axis=0,
                                                                                             ddof=1)
    VAR = np.sqrt((hidden_treated_var + hidden_controlled_var) / 2)
    # hidden_deviation = np.abs(hidden_treated_mu - hidden_controlled_mu) / VAR
    # hidden_deviation[np.isnan(hidden_deviation)] = 0  # -1  # 0  # float('-inf') represent VAR is 0
    hidden_deviation = np.divide(
        np.abs(hidden_treated_mu - hidden_controlled_mu),
        VAR, out=np.zeros_like(hidden_treated_mu), where=VAR != 0)

    max_unbalanced_original = np.max(hidden_deviation)

    # Weighted SMD
    hidden_treated_w_mu, hidden_treated_w_var = weighted_mean(hidden_treated, treated_w), weighted_var(hidden_treated,
                                                                                                       treated_w)
    hidden_controlled_w_mu, hidden_controlled_w_var = weighted_mean(hidden_controlled, controlled_w), weighted_var(
        hidden_controlled, controlled_w)
    VAR_w = np.sqrt((hidden_treated_w_var + hidden_controlled_w_var) / 2)
    # hidden_deviation_w = np.abs(hidden_treated_w_mu - hidden_controlled_w_mu) / VAR_w
    # hidden_deviation_w[np.isnan(hidden_deviation_w)] = 0  # -1  # 0
    hidden_deviation_w = np.divide(
        np.abs(hidden_treated_w_mu - hidden_controlled_w_mu),
        VAR_w, out=np.zeros_like(hidden_treated_w_mu), where=VAR_w != 0)

    max_unbalanced_weighted = np.max(hidden_deviation_w)

    return max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w

class PropensityEstimator:
    def __init__(self, learner, criteria, paras_grid=None, random_seed=0, print_result=True):
        self.learner = learner
        self.random_seed = random_seed
        assert self.learner in ('LR', 'MLP', 'XGBOOST', 'LIGHTGBM')

        self.criteria = criteria
        self.print_result = print_result


        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            self.paras_grid = {}
        else:
            self.paras_grid = {k: v for k, v in paras_grid.items()}
            for k, v in self.paras_grid.items():
                if isinstance(v, str) or not isinstance(v, (list, set, np.ndarray, pd.Series)):
                    if self.print_result:
                        print(k, v, 'is a fixed parameter')
                    self.paras_grid[k] = [v, ]

        if self.paras_grid:
            paras_names, paras_v = zip(*self.paras_grid.items())
            paras_list = list(itertools.product(*paras_v))
            self.paras_names = paras_names
            self.paras_list = [{self.paras_names[i]: para[i] for i in range(len(para))} for para in paras_list]
            if self.learner == 'LR':
                no_penalty_case = {'penalty': None, 'max_iter': 200, 'random_state': random_seed}
                if (no_penalty_case not in self.paras_list) and (len(self.paras_list) > 1):
                    # self.paras_list.append(no_penalty_case)
                    self.paras_list = [no_penalty_case, ] + self.paras_list  # debug
                    if self.print_result:
                        print('Add no penalty case to logistic regression model:', no_penalty_case)
        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None

        self.best_hyper_paras_nestcv = []
        self.best_model_nestcv = []

        self.best_val = float('-inf')
        self.best_balance = float('inf')
        self.best_loss = float('inf')

        self.best_val_nestcv = []
        self.best_balance_nestcv = []

        self.global_best_val = float('-inf')
        self.global_best_balance = float('inf')
        self.global_best_loss = float('inf')

        self.best_balance_k_folds_detail = []  # k #(SMD>threshold)
        self.best_val_k_folds_detail = []  # k AUC
        self.best_loss_k_folds_detail = []

        self.best_balance_k_folds_detail_nestcv = []  # k #(SMD>threshold)
        self.best_val_k_folds_detail_nestcv = []  # k AUC

        self.results = []
        self.results_retrain = []
        self.results_agg = []

    @staticmethod
    def _evaluation_helper(X, T, T_pre):
        loss = log_loss(T, T_pre)
        auc = roc_auc_score(T, T_pre)
        max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, T_pre, normalized=True, verbose=False)
        n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
        n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
        result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
        return result

    # @staticmethod
    # def _evaluation_effect_helper(X, T, T_pre, Y, verbose=1):
    #     balance_result = PropensityEstimator._evaluation_helper(X, T, T_pre)
    #     tkm = model_eval_common_simple(
    #         X, T, Y, T_pre, loss=np.nan, verbose=verbose,
    #         normalized=True, figsave='')
    #     result = (
    #     tkm[2][0], tkm[2][1][0], tkm[2][1][1], tkm[2][2].summary.p.treatment if pd.notna(tkm[2][2]) else np.nan,
    #     tkm[3][0], tkm[3][1][0], tkm[3][1][1], tkm[3][2].summary.p.treatment if pd.notna(tkm[3][2]) else np.nan)
    #     # label = ['HR_ori', 'HR_ori_CI_lower', 'HR_ori_CI_upper', 'HR_ori_p',
    #     #          'HR_IPTW', 'HR_IPTW_CI_lower', 'HR_IPTW_CI_upper','HR_IPTW_p', ]
    #     # label = [prefix+x for x in label]
    #
    #     return balance_result + result

    def fit(self, X_train, T_train, X_val, T_val, verbose=1):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        i = -1
        for para_d in tqdm(self.paras_list):
            i += 1
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                else:
                    para_d['solver'] = 'lbfgs'
                model = LogisticRegression(**para_d).fit(X_train, T_train)
            elif self.learner == 'MLP':
                model = MLPClassifier(**para_d).fit(X_train, T_train)
            # elif self.learner == 'XGBOOST':
            #     model = xgb.XGBClassifier(**para_d).fit(X_train, T_train)
            # elif self.learner == 'LIGHTGBM':
            #     model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
            else:
                raise ValueError

            T_train_pre = model.predict_proba(X_train)[:, 1]
            T_val_pre = model.predict_proba(X_val)[:, 1]

            result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
            result_val = self._evaluation_helper(X_val, T_val, T_val_pre)

            result_trainval = self._evaluation_helper(
                np.concatenate((X_train, X_val)),
                np.concatenate((T_train, T_val)),
                np.concatenate((T_train_pre, T_val_pre))
            )

            self.results.append((i, para_d) + result_train + result_val + result_trainval)
            # might use [0] loss in the future
            if (result_trainval[5] < self.best_balance) or \
                    ((result_trainval[5] == self.best_balance) and (result_val[1] > self.best_val)):
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = result_val[1]
                self.best_balance = result_trainval[5]

            if result_val[1] > self.global_best_val:
                self.global_best_val = result_val[1]

            if result_trainval[5] <= self.global_best_balance:
                self.global_best_balance = result_trainval[5]

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'paras'] + [pre + x for pre in ['train_', 'val_', 'trainval_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def fit_and_test(self, X_train, T_train, X_val, T_val, X_test, T_test, verbose=1):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        i = -1
        for para_d in tqdm(self.paras_list):
            i += 1
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                else:
                    para_d['solver'] = 'lbfgs'
                model = LogisticRegression(**para_d).fit(X_train, T_train)
            elif self.learner == 'MLP':
                model = MLPClassifier(**para_d).fit(X_train, T_train)
            # elif self.learner == 'LIGHTGBM':
            #     model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
            else:
                raise ValueError

            T_train_pre = model.predict_proba(X_train)[:, 1]
            T_val_pre = model.predict_proba(X_val)[:, 1]
            T_test_pre = model.predict_proba(X_test)[:, 1]

            result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
            result_val = self._evaluation_helper(X_val, T_val, T_val_pre)
            result_test = self._evaluation_helper(X_test, T_test, T_test_pre)

            result_trainval = self._evaluation_helper(
                np.concatenate((X_train, X_val)),
                np.concatenate((T_train, T_val)),
                np.concatenate((T_train_pre, T_val_pre))
            )

            result_all = self._evaluation_helper(
                np.concatenate((X_train, X_val, X_test)),
                np.concatenate((T_train, T_val, T_test)),
                np.concatenate((T_train_pre, T_val_pre, T_test_pre))
            )

            self.results.append((i, para_d) + result_train + result_val + result_test + result_trainval + result_all)

            if (result_trainval[5] < self.best_balance) or \
                    ((result_trainval[5] == self.best_balance) and (result_val[1] > self.best_val)):
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = result_val[1]
                self.best_balance = result_trainval[5]

            if result_val[1] > self.global_best_val:
                self.global_best_val = result_val[1]

            if result_trainval[5] <= self.global_best_balance:
                self.global_best_balance = result_trainval[5]

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'paras'] + [pre + x for pre in ['train_', 'val_', 'test_', 'trainval_', 'all_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def _model_estimation(self, para_d, X_train, T_train):
        # model estimation on training data
        if self.learner == 'LR':
            if para_d.get('penalty', '') == 'l1':
                para_d['solver'] = 'liblinear'
            else:
                para_d['solver'] = 'lbfgs'
            model = LogisticRegression(**para_d).fit(X_train, T_train)
        elif self.learner == 'MLP':
            model = MLPClassifier(**para_d).fit(X_train, T_train)
        else:
            raise ValueError

        return model

    def cross_validation_fit(self, X, T, kfold=5, verbose=1, shuffle=True):
        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            if self.print_result:
                print('Model {} Searching Space N={} by '
                      '{}-k-fold cross validation: '.format(self.learner,
                                                            len(self.paras_list),
                                                            kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)
        T = np.asarray(T)
        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            i_model_loss_over_kfold = []
            for k, (train_index, test_index) in enumerate(kf.split(X), 1):
                if self.print_result:
                    print('Training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                X_test = X[test_index, :]
                T_test = T[test_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba(X_train)[:, 1]
                T_test_pre = model.predict_proba(X_test)[:, 1]

                # evaluating goodness-of-balance and goodness-of-fit
                result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
                result_all = self._evaluation_helper(
                    np.concatenate((X_train, X_test)),
                    np.concatenate((T_train, T_test)),
                    np.concatenate((T_train_pre, T_test_pre))
                )  # (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                i_model_balance_over_kfold.append(result_all[5])
                i_model_fit_over_kfold.append(result_test[1])
                i_model_loss_over_kfold.append(result_test[0])

                self.results.append((i, k, para_d) + result_train + result_test + result_all)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]
            i_model_loss = [np.mean(i_model_loss_over_kfold), np.std(i_model_loss_over_kfold)]

            if self.criteria == "balance":
                if (i_model_balance[0] < self.best_balance) or \
                        ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
                # if i_model_fit[0] > self.best_val:
                    # model with current best configuration re-trained on the whole dataset.
                    # self.best_model = self._model_estimation(para_d, X, T)
                    self.best_hyper_paras = para_d
                    self.best_balance = i_model_balance[0]
                    self.best_val = i_model_fit[0]
                    self.best_loss = i_model_loss[0]
                    self.best_balance_k_folds_detail = i_model_balance_over_kfold
                    self.best_val_k_folds_detail = i_model_fit_over_kfold
                    self.best_loss_k_folds_detail = i_model_loss_over_kfold
            elif self.criteria == "auroc":
                if i_model_fit[0] > self.best_val:
                # if i_model_fit[0] > self.best_val:
                    # model with current best configuration re-trained on the whole dataset.
                    # self.best_model = self._model_estimation(para_d, X, T)
                    self.best_hyper_paras = para_d
                    self.best_balance = i_model_balance[0]
                    self.best_val = i_model_fit[0]
                    self.best_loss = i_model_loss[0]
                    self.best_balance_k_folds_detail = i_model_balance_over_kfold
                    self.best_val_k_folds_detail = i_model_fit_over_kfold
                    self.best_loss_k_folds_detail = i_model_loss_over_kfold
            elif self.criteria == "loss":
                if i_model_loss[0] < self.best_loss:
                # if i_model_fit[0] > self.best_val:
                    # model with current best configuration re-trained on the whole dataset.
                    # self.best_model = self._model_estimation(para_d, X, T)
                    self.best_hyper_paras = para_d
                    self.best_balance = i_model_balance[0]
                    self.best_val = i_model_fit[0]
                    self.best_loss = i_model_loss[0]
                    self.best_balance_k_folds_detail = i_model_balance_over_kfold
                    self.best_val_k_folds_detail = i_model_fit_over_kfold
                    self.best_loss_k_folds_detail = i_model_loss_over_kfold
            else:
                raise NotImplementedError

            # if (i_model_balance[0] < self.best_balance) or \
            #         ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
            #     # model with current best configuration re-trained on the whole dataset.
            #     # self.best_model = self._model_estimation(para_d, X, T)
            #     self.best_hyper_paras = para_d
            #     self.best_balance = i_model_balance[0]
            #     self.best_val = i_model_fit[0]
            #     self.best_balance_k_folds_detail = i_model_balance_over_kfold
            #     self.best_val_k_folds_detail = i_model_fit_over_kfold

            if i_model_fit[0] > self.global_best_val:
                self.global_best_val = i_model_fit[0]

            if i_model_balance[0] < self.global_best_balance:
                self.global_best_balance = i_model_balance[0]

            if i_model_loss[0] < self.global_best_loss:
                self.global_best_loss = i_model_loss[0]

            # save re-trained results on the whole data, for model selection exp only. Not necessary for later use
            model_retrain = self._model_estimation(para_d, X, T)
            T_pre = model_retrain.predict_proba(X)[:, 1]
            result_retrain = self._evaluation_helper(X, T, T_pre)
            self.results_retrain.append((i, 'retrain', para_d) + result_retrain)

            if verbose:
                if self.print_result:
                    self.report_stats()

        # end of training
        if self.print_result:
            print('best model parameter:', self.best_hyper_paras)
            print('re-training best model on all the data using best model parameter...')
        self.best_model = self._model_estimation(self.best_hyper_paras, X, T)
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain all_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        # results_agg = self.results.groupby('paras_str').agg(['mean', 'std']).reset_index().sort_values(
        #     by=[('i', 'mean')])
        # results_agg.columns = results_agg.columns.to_flat_index()
        # results_agg.columns = results_agg.columns.map('-'.join)
        # self.results_agg = pd.merge(results_agg, self.results_retrain, left_on='paras_str-', right_on='paras_str',
        #                             how='left')
        #
        # if verbose:
        #     self.report_stats()
        if self.print_result:
            print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def cross_validation_fit_withtestset(self, X, T, X_test, T_test, kfold=10, verbose=1, shuffle=True):
        """
        # CV model selection and training on X, T
        # out-of-sample test on the Xtest and Ttest
        :return:
        """

        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            print('Model {} Searching Space N={} by '
                  '{}-k-fold cross validation: '.format(self.learner,
                                                        len(self.paras_list),
                                                        kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)  # as training set for cross-valiadtion into train and val
        T = np.asarray(T)  # as training set for cross-valiadtion into train and val
        # for out-of-sample test
        X_test = np.asarray(X_test)
        T_test = np.asarray(T_test)
        X_all = np.concatenate((X, X_test))
        T_all = np.concatenate((T, T_test))

        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            for k, (train_index, val_index) in enumerate(kf.split(X), 1):
                print('Training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                X_val = X[val_index, :]
                T_val = T[val_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba(X_train)[:, 1]
                T_val_pre = model.predict_proba(X_val)[:, 1]

                # evaluating goodness-of-balance and goodness-of-fit
                result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                result_val = self._evaluation_helper(X_val, T_val, T_val_pre)
                result_trainval = self._evaluation_helper(
                    np.concatenate((X_train, X_val)),
                    np.concatenate((T_train, T_val)),
                    np.concatenate((T_train_pre, T_val_pre))
                )  # (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                i_model_balance_over_kfold.append(result_trainval[5])
                i_model_fit_over_kfold.append(result_val[1])

                self.results.append((i, k, para_d) + result_train + result_val + result_trainval)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]

            if (i_model_balance[0] < self.best_balance) or \
                    ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
                # model with current best configuration re-trained on the whole dataset.
                # self.best_model = self._model_estimation(para_d, X, T)
                self.best_hyper_paras = para_d
                self.best_balance = i_model_balance[0]
                self.best_val = i_model_fit[0]
                self.best_balance_k_folds_detail = i_model_balance_over_kfold
                self.best_val_k_folds_detail = i_model_fit_over_kfold

            if i_model_fit[0] > self.global_best_val:
                self.global_best_val = i_model_fit[0]

            if i_model_balance[0] < self.global_best_balance:
                self.global_best_balance = i_model_balance[0]

            # save re-trained results on the whole (training+val) data, for model selection exp only. Not necessary for later use
            model_retrain = self._model_estimation(para_d, X, T)
            T_pre = model_retrain.predict_proba(X)[:, 1]
            result_retrain = self._evaluation_helper(X, T, T_pre)

            # testing model on the test data, for model selection exp only. Not necessary for later use
            T_test_pre = model_retrain.predict_proba(X_test)[:, 1]
            result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
            T_all_pre = model_retrain.predict_proba(X_all)[:, 1]
            result_all = self._evaluation_helper(X_all, T_all, T_all_pre)

            # cross-validation part build train and val results
            # this part build retrain on train+val, test, and all results.
            self.results_retrain.append((i, 'retrain', para_d) + result_retrain + result_test + result_all)

            if verbose:
                print('Finish training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list),
                                                                                           para_d, kfold))
                print('CV Balance mean, std:', i_model_balance, 'k_folds:', i_model_balance_over_kfold)
                print('CV Fit mean, std:', i_model_fit, 'k_folds:', i_model_fit_over_kfold)
                self.report_stats()

        # end of training
        print('best model parameter:', self.best_hyper_paras)
        print('re-training best model on all the data using best model parameter...')
        # best model is used in predicting ps
        # retrained here
        self.best_model = self._model_estimation(self.best_hyper_paras, X, T)
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain trainval_'] for x in
                                               name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['trainval_', 'test_', 'all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        results_agg = self.results.groupby('paras_str').agg(['mean', 'std']).reset_index().sort_values(
            by=[('i', 'mean')])
        results_agg.columns = results_agg.columns.to_flat_index()
        results_agg.columns = results_agg.columns.map('-'.join)
        self.results_agg = pd.merge(results_agg, self.results_retrain, left_on='paras_str-', right_on='paras_str',
                                    how='left')

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def evaluate_test(self, X_test, T_test, X_test_all):
        # propensity scores on training and testing datasets
        T_test_pre = self.best_model.predict_proba(X_test)[:, 1]

        # evaluating goodness-of-balance and goodness-of-fit
        result_test = self._evaluation_helper(X_test_all, T_test, T_test_pre)

        return result_test

    def cross_validation_fit_withtestset_witheffect(self, X, T, Y, X_test, T_test, Y_test, kfold=10, verbose=1,
                                                    shuffle=True):
        """
        # CV model selection and training on X, T
        # out-of-sample test on the Xtest and Ttest
        :return:
        """

        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            print('Model {} Searching Space N={} by '
                  '{}-k-fold cross validation: '.format(self.learner,
                                                        len(self.paras_list),
                                                        kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)  # as training set for cross-valiadtion into train and val
        T = np.asarray(T)  # as training set for cross-valiadtion into train and val
        Y = np.asarray(Y)
        # for out-of-sample test
        X_test = np.asarray(X_test)
        T_test = np.asarray(T_test)
        Y_test = np.asarray(Y_test)

        X_all = np.concatenate((X, X_test))
        T_all = np.concatenate((T, T_test))
        Y_all = np.concatenate((Y, Y_test))

        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            for k, (train_index, val_index) in enumerate(kf.split(X), 1):
                print('Training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                Y_train = Y[train_index]
                X_val = X[val_index, :]
                T_val = T[val_index]
                Y_val = Y[val_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba(X_train)[:, 1]
                T_val_pre = model.predict_proba(X_val)[:, 1]

                # evaluating goodness-of-balance and goodness-of-fit
                result_train = self._evaluation_effect_helper(X_train, T_train, T_train_pre, Y_train, verbose=0)
                result_val = self._evaluation_effect_helper(X_val, T_val, T_val_pre, Y_val, verbose=0)
                result_trainval = self._evaluation_effect_helper(
                    np.concatenate((X_train, X_val)),
                    np.concatenate((T_train, T_val)),
                    np.concatenate((T_train_pre, T_val_pre)),
                    np.concatenate((Y_train, Y_val)), verbose=0
                )  # (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                i_model_balance_over_kfold.append(result_trainval[5])
                i_model_fit_over_kfold.append(result_val[1])

                self.results.append((i, k, para_d) + result_train + result_val + result_trainval)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]

            if (i_model_balance[0] < self.best_balance) or \
                    ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
                # model with current best configuration re-trained on the whole dataset.
                # self.best_model = self._model_estimation(para_d, X, T)
                self.best_hyper_paras = para_d
                self.best_balance = i_model_balance[0]
                self.best_val = i_model_fit[0]
                self.best_balance_k_folds_detail = i_model_balance_over_kfold
                self.best_val_k_folds_detail = i_model_fit_over_kfold

            if i_model_fit[0] > self.global_best_val:
                self.global_best_val = i_model_fit[0]

            if i_model_balance[0] < self.global_best_balance:
                self.global_best_balance = i_model_balance[0]

            # save re-trained results on the whole (training+val) data, for model selection exp only. Not necessary for later use
            model_retrain = self._model_estimation(para_d, X, T)
            T_pre = model_retrain.predict_proba(X)[:, 1]
            print('........results on training')
            result_retrain = self._evaluation_effect_helper(X, T, T_pre, Y)

            # testing model on the test data, for model selection exp only. Not necessary for later use
            T_test_pre = model_retrain.predict_proba(X_test)[:, 1]
            print('........results on test')
            result_test = self._evaluation_effect_helper(X_test, T_test, T_test_pre, Y_test)
            T_all_pre = model_retrain.predict_proba(X_all)[:, 1]
            print('........results on all')
            result_all = self._evaluation_effect_helper(X_all, T_all, T_all_pre, Y_all)

            # cross-validation part build train and val results
            # this part build retrain on train+val, test, and all results.
            self.results_retrain.append((i, 'retrain', para_d) + result_retrain + result_test + result_all)

            if verbose:
                print('Finish training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list),
                                                                                           para_d, kfold))
                print('CV Balance mean, std:', i_model_balance, 'k_folds:', i_model_balance_over_kfold)
                print('CV Fit mean, std:', i_model_fit, 'k_folds:', i_model_fit_over_kfold)
                self.report_stats()

        # end of training
        print('best model parameter:', self.best_hyper_paras)
        print('re-training best model on all the data using best model parameter...')
        # best model is used in predicting ps
        # retrained here
        self.best_model = self._model_estimation(self.best_hyper_paras, X, T)
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw',
                'HR_ori', 'HR_ori_CI_lower', 'HR_ori_CI_upper', 'HR_ori_p',
                'HR_IPTW', 'HR_IPTW_CI_lower', 'HR_IPTW_CI_upper', 'HR_IPTW_p']
        col_name = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain trainval_'] for x in
                                               name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['trainval_', 'test_', 'all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        results_agg = self.results.groupby('paras_str').agg(['mean', 'std']).reset_index().sort_values(
            by=[('i', 'mean')])
        results_agg.columns = results_agg.columns.to_flat_index()
        results_agg.columns = results_agg.columns.map('-'.join)
        self.results_agg = pd.merge(results_agg, self.results_retrain, left_on='paras_str-', right_on='paras_str',
                                    how='left')

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def nested_cross_validation_fit(self, X, T, kfold_out=10, kfold_in=5, verbose=1, shuffle=True):
        """
        Nested cv schema
        # CV model selection and training on X, T
        # out-of-sample test on the Xtest and Ttest
        :return:
        """
        start_time = time.time()

        self.best_hyper_paras_nestcv = [None, ] * kfold_out
        self.best_model_nestcv = [None, ] * kfold_out
        self.best_val_nestcv = [float('-inf'), ] * kfold_out
        self.best_balance_nestcv = [float('inf'), ] * kfold_out
        self.best_balance_k_folds_detail_nestcv = [None, ] * kfold_out
        self.best_val_k_folds_detail_nestcv = [None, ] * kfold_out

        kf_out = KFold(n_splits=kfold_out, random_state=self.random_seed, shuffle=shuffle)
        kf_in = KFold(n_splits=kfold_in, random_state=self.random_seed, shuffle=shuffle)

        if verbose:
            print('Model {} Searching Space N={} by Out {}-k-fold IN {}-fold nested cross validation: '.format(
                self.learner, len(self.paras_list), kf_out.get_n_splits(), kf_in.get_n_splits()),
                self.paras_grid)

        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance

        X = np.asarray(X)  # as training set for cross-valiadtion into train and val
        T = np.asarray(T)  # as training set for cross-valiadtion into train and val
        # for out-of-sample test
        # X_test = np.asarray(X_test)
        # T_test = np.asarray(T_test)
        # X_all = np.concatenate((X, X_test))
        # T_all = np.concatenate((T, T_test))
        for kout, (trainval_index, test_index) in tqdm(enumerate(kf_out.split(X), 0), total=kfold_out):
            X_trainval = X[trainval_index, :]
            T_trainval = T[trainval_index]
            X_test = X[test_index, :]
            T_test = T[test_index]

            # what else results need to store?
            for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
                i_model_balance_over_kfold = []
                i_model_fit_over_kfold = []
                for kin, (train_index, val_index) in enumerate(kf_in.split(X_trainval), 0):
                    print('{}-th out fold, training {}th (/{}) model {} over the {}th-in-fold data'.format(
                        kout, i, len(self.paras_list), para_d, kin))
                    # training and testing datasets:
                    X_train = X_trainval[train_index, :]
                    T_train = T_trainval[train_index]
                    X_val = X_trainval[val_index, :]
                    T_val = T_trainval[val_index]

                    # model estimation on training data
                    model = self._model_estimation(para_d, X_train, T_train)

                    # propensity scores on training and testing datasets
                    T_train_pre = model.predict_proba(X_train)[:, 1]
                    T_val_pre = model.predict_proba(X_val)[:, 1]

                    # evaluating goodness-of-balance and goodness-of-fit
                    result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                    result_val = self._evaluation_helper(X_val, T_val, T_val_pre)
                    result_trainval = self._evaluation_helper(
                        np.concatenate((X_train, X_val)),
                        np.concatenate((T_train, T_val)),
                        np.concatenate((T_train_pre, T_val_pre))
                    )  # (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                    i_model_balance_over_kfold.append(result_trainval[5])
                    i_model_fit_over_kfold.append(result_val[1])

                    self.results.append((kout, i, kin, para_d) + result_train + result_val + result_trainval)
                    # end of one fold

                i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
                i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]

                if (i_model_balance[0] < self.best_balance_nestcv[kout]) or \
                        ((i_model_balance[0] == self.best_balance_nestcv[kout]) and (i_model_fit[0] > self.best_val_nestcv[kout])):
                    # model with current best configuration re-trained on the whole dataset.
                    # self.best_model = self._model_estimation(para_d, X, T)
                    # we can keep these codes, just global best over k-out fold.
                    # However, this is also depends on sampled datasets, which might be easier to balance
                    self.best_hyper_paras_nestcv[kout] = para_d
                    self.best_balance_nestcv[kout] = i_model_balance[0]
                    self.best_val_nestcv[kout] = i_model_fit[0]
                    self.best_balance_k_folds_detail_nestcv[kout] = i_model_balance_over_kfold
                    self.best_val_k_folds_detail_nestcv[kout] = i_model_fit_over_kfold


                if i_model_fit[0] > self.global_best_val: # global best is not useful here
                    self.global_best_val = i_model_fit[0]

                if i_model_balance[0] < self.global_best_balance: # global best is not useful here
                    self.global_best_balance = i_model_balance[0]

                # save re-trained results on the training+val data, for model selection exp only. Not necessary for later use
                model_retrain = self._model_estimation(para_d, X_trainval, T_trainval)
                T_trainval_pre = model_retrain.predict_proba(X_trainval)[:, 1]
                result_retrain = self._evaluation_helper(X_trainval, T_trainval, T_trainval_pre)

                # testing model on the test data, for model selection exp only. Not necessary for later use
                T_test_pre = model_retrain.predict_proba(X_test)[:, 1]
                result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
                T_all_pre = model_retrain.predict_proba(X)[:, 1]
                result_all = self._evaluation_helper(X, T, T_all_pre)

                # cross-validation part build train and val results
                # this part build retrain on train+val, test, and all results.
                self.results_retrain.append((kout, i, 'retrain on trainval', para_d) + result_retrain + result_test + result_all)

                if verbose:
                    print('Finish training {}th-Out-fold {}th (/{}) model {} over the {}th-In-fold data'.format(
                        kout, i, len(self.paras_list), para_d, kin))
                    print('CV Balance mean, std:', i_model_balance, 'k_folds:', i_model_balance_over_kfold)
                    print('CV Fit mean, std:', i_model_fit, 'k_folds:', i_model_fit_over_kfold)
                    self.report_stats()

            # end of training
            print('best model parameter in kout {}:'.format(kout), self.best_hyper_paras_nestcv[kout])
            print('re-training best model on all the data using best model parameter...')
            # best model is used in predicting ps
            # retrained here
            # should we keep all k-fold model, or just the global best?
            self.best_model_nestcv[kout] = self._model_estimation(self.best_hyper_paras_nestcv[kout], X, T)

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['fold-k-out', 'i', 'fold-k-in', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain trainval_'] for x in
                                               name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['fold-k-out', 'i', 'fold-k-in', 'paras'] + [pre + x for pre in ['trainval_', 'test_', 'all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        results_agg = self.results.drop(columns=['paras']).groupby(['fold-k-out', 'paras_str']).agg(['mean', 'std']).reset_index().sort_values(
            by=[('fold-k-out', ''), ('i', 'mean')])
        results_agg.columns = results_agg.columns.to_flat_index()
        results_agg.columns = results_agg.columns.map('-'.join)
        self.results_agg = pd.merge(results_agg, self.results_retrain,
                                    left_on=['fold-k-out-', 'paras_str-'], right_on=['fold-k-out', 'paras_str'],
                                    how='left')

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def report_stats(self):
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best balance: ', self.best_balance, ' Global Best balance: ', self.global_best_balance)
        print('Best fit value ', self.best_val, ' Global Best fit balue: ', self.global_best_val)
        try:
            pd.set_option('display.max_columns', None)
            describe = self.results.describe()
            print('AUC stats:\n', describe)
            return describe
        except:
            print('')

    def predict_ps(self, X):
        pred_ps = self.best_model.predict_proba(X)[:, 1]
        # pred_clip_propensity = np.clip(pred_propensity, a_min=np.quantile(pred_propensity, 0.1), a_max=np.quantile(pred_propensity, 0.9))
        return pred_ps

    def predict_loss(self, X, T):
        T_pre = self.predict_ps(X)
        return log_loss(T, T_pre)

    def predict_ps_nestedCV(self, X, kout):
        pred_ps = self.best_model_nestcv[kout].predict_proba(X)[:, 1]
        return pred_ps

    def predict_loss_nestedCV(self, X, T, kout):
        T_pre = self.predict_ps_nestedCV(X, kout)
        return log_loss(T, T_pre)

class FeatureSelector:
    def __init__(self, data, a_index, y_index, learner, sel_criteria, cv_criteria, p_thre, random_seed=0):
        self.data = data
        self.a_index = a_index
        self.y_index = y_index
        self.learner = learner
        self.sel_criteria = sel_criteria
        self.cv_criteria = cv_criteria
        self.p_thre = p_thre
        self.random_seed = random_seed
        self.result_dict = {}

    def cond_ind_test(self, a_idx, b_idx, c_idx):
        idx_set1 = np.array(list(c_idx))
        idx_set2 = np.array(list(b_idx) + list(c_idx))
        idx_set1 = np.sort(idx_set1)
        idx_set2 = np.sort(idx_set2)
        str_set1 = str(a_idx) + ":" + "#".join([str(item) for item in idx_set1])
        str_set2 = str(a_idx) + ":" + "#".join([str(item) for item in idx_set2])

        if self.learner == "LR":
            paras_grid = {
                'penalty': ['l1', 'l2'],
                'C': 10 ** np.arange(-3, 3, 0.5),
                # 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
                'max_iter': [200],  # [100, 200, 500],
                'random_state': [0],
            }
        elif self.learner == "MLP":
            paras_grid = {
                'alpha': 10 ** np.arange(-1, 2, 0.25),
            }
        else:
            raise NotImplementedError

        if len(idx_set1) > 0 and str_set1 not in self.result_dict:
            ps = PropensityEstimator(learner=self.learner, paras_grid=paras_grid, criteria=self.cv_criteria)
            ps.cross_validation_fit(self.data[:, idx_set1], self.data[:, a_idx])
            self.result_dict[str_set1] = {'model':ps.best_model, 'loss':ps.best_loss}
            logloss_set1 = ps.best_loss
        elif len(idx_set1) == 0:
            a = self.data[:, a_idx]
            p = a.mean()
            logloss_set1 = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            logloss_set1 = self.result_dict[str_set1]['loss']

        if str_set2 not in self.result_dict:
            ps = PropensityEstimator(learner=self.learner, paras_grid=paras_grid, criteria=self.cv_criteria, print_result=False)
            ps.cross_validation_fit(self.data[:, idx_set2], self.data[:, a_idx])
            self.result_dict[str_set1] = {'model':ps.best_model, 'loss':ps.best_loss}
            logloss_set2 = ps.best_loss
        else:
            logloss_set2 = self.result_dict[str_set1]['loss']

        return logloss_set1, logloss_set2

    def fit_test(self, a_idx, c_idx):
        idx_set1 = np.array(list(c_idx))
        idx_set1 = np.sort(idx_set1)
        str_set1 = str(a_idx) + ":" + "#".join([str(item) for item in idx_set1])

        if self.learner == "LR":
            paras_grid = {
                'penalty': ['l1', 'l2'],
                'C': 10 ** np.arange(-3, 3, 0.5),
                # 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
                'max_iter': [200],  # [100, 200, 500],
                'random_state': [0],
            }
        elif self.learner == "MLP":
            paras_grid = {
                'alpha': 10 ** np.arange(-1, 2, 0.25),
            }
        else:
            raise NotImplementedError

        if len(idx_set1) > 0 and str_set1 not in self.result_dict:
            ps = PropensityEstimator(learner=self.learner, paras_grid=paras_grid, criteria=self.cv_criteria, print_result=False)
            ps.cross_validation_fit(self.data[:, idx_set1], self.data[:, a_idx])
            self.result_dict[str_set1] = {'model':ps.best_model, 'loss':ps.best_loss}
            logloss_set1 = ps.best_loss
        elif len(idx_set1) == 0:
            a = self.data[:, a_idx]
            p = a.mean()
            logloss_set1 = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            logloss_set1 = self.result_dict[str_set1]['loss']

        return logloss_set1

    def search(self, target_idx, prior_list):
        best_set = [prior_list[0]]
        best_loss = self.fit_test(target_idx, best_set)
        for i in range(1, len(prior_list)):
            cur_idx = prior_list[i]
            cur_set = deepcopy(best_set) + [cur_idx]
            cur_loss = self.fit_test(target_idx, cur_set)

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_set = deepcopy(cur_set)

            print(i, best_set, best_loss)
        return best_set, best_loss

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim = 100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def predict_proba_np(self, x):
        X_tensor = torch.Tensor(x)
        outputs = self.forward(X_tensor)
        pred_prob = torch.sigmoid(outputs)

        return pred_prob.detach().numpy()[:, 0]


class LinearModel(nn.Module):
    def __init__(self, in_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        out = self.fc(x)
        return torch.sigmoid(out)

    def predict_proba_np(self, x):
        X_tensor = torch.Tensor(x)
        outputs = self.forward(X_tensor)
        pred_prob = torch.sigmoid(outputs)

        return pred_prob.detach().numpy()[:, 0]

class CoxphModel(nn.Module):
    def __init__(self, in_dim):
        super(CoxphModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1)

    def forward(self, x):
        out = self.fc1(x)
        return out


class PropensityEstimatorTorch:
    def __init__(self, criteria, paras_grid=None, random_seed=0, print_result=True):
        self.random_seed = random_seed
        self.criteria = criteria
        self.print_result = print_result

        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            self.paras_grid = {}
        else:
            self.paras_grid = {k: v for k, v in paras_grid.items()}
            for k, v in self.paras_grid.items():
                if isinstance(v, str) or not isinstance(v, (list, set, np.ndarray, pd.Series)):
                    if self.print_result:
                        print(k, v, 'is a fixed parameter')
                    self.paras_grid[k] = [v, ]

        if self.paras_grid:
            paras_names, paras_v = zip(*self.paras_grid.items())
            paras_list = list(itertools.product(*paras_v))
            self.paras_names = paras_names
            self.paras_list = [{self.paras_names[i]: para[i] for i in range(len(para))} for para in paras_list]
        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None

        self.best_hyper_paras_nestcv = []
        self.best_model_nestcv = []

        self.best_val = float('-inf')
        self.best_balance = float('inf')
        self.best_loss = float('inf')

        self.best_val_nestcv = []
        self.best_balance_nestcv = []

        self.global_best_val = float('-inf')
        self.global_best_balance = float('inf')
        self.global_best_loss = float('inf')

        self.best_balance_k_folds_detail = []  # k #(SMD>threshold)
        self.best_val_k_folds_detail = []  # k AUC
        self.best_loss_k_folds_detail = []

        self.best_balance_k_folds_detail_nestcv = []  # k #(SMD>threshold)
        self.best_val_k_folds_detail_nestcv = []  # k AUC

        self.results = []
        self.results_retrain = []
        self.results_agg = []

    @staticmethod
    def _evaluation_helper(X, T, T_pre):
        loss = log_loss(T, T_pre)
        auc = roc_auc_score(T, T_pre)
        max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, T_pre, normalized=True, verbose=False)
        n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
        n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
        result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
        return result

    def _model_estimation(self, para_d, X_train, T_train):
        model = MLP(X_train.shape[1], para_d['hidden_dim'])
        optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=para_d['weight_decay'])

        if 'max_iter' in para_d:
            max_iter = para_d['max_iter']
        else:
            max_iter = 5000

        # train_set = CustomDataset(X_train, T_train)
        # train_loader = DataLoader(train_set, batch_size=200, drop_last=False, shuffle=True)
        #
        # model.train()
        # for iter in range(max_iter + 1):
        #     for (batch_X, batch_T) in train_loader:
        #         optim.zero_grad()
        #         outputs = model.forward(batch_X)
        #         pred_prob = torch.sigmoid(outputs)
        #         loss = (- batch_T * torch.log(pred_prob[:, 0] + 1e-8) - (1 - batch_T) * torch.log((1 - pred_prob[:, 0]) + 1e-8)).mean()
        #         loss.backward()
        #         optim.step()

        X_tensor, T_tensor = torch.Tensor(X_train), torch.Tensor(T_train)
        model.train()
        for iter in range(max_iter + 1):
            optim.zero_grad()
            outputs = model.forward(X_tensor)
            pred_prob = torch.sigmoid(outputs)
            loss = (- T_tensor * torch.log(pred_prob[:, 0] + 1e-8) - (1 - T_tensor) * torch.log((1 - pred_prob[:, 0]) + 1e-8)).mean()
            loss.backward()
            optim.step()
            if self. print_result:
                if iter % 1000 == 0:
                    print(iter, loss)
        model.eval()
        return model

    def cross_validation_fit(self, X, T, kfold=5, verbose=1, shuffle=True):
        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            if self.print_result:
                print('Searching Space N={} by '
                      '{}-k-fold cross validation: '.format(len(self.paras_list),
                                                            kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)
        T = np.asarray(T)
        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            i_model_loss_over_kfold = []
            for k, (train_index, test_index) in enumerate(kf.split(X), 1):
                if self.print_result:
                    print('Training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                X_test = X[test_index, :]
                T_test = T[test_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba_np(X_train)
                T_test_pre = model.predict_proba_np(X_test)

                # evaluating goodness-of-balance and goodness-of-fit
                result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
                result_all = self._evaluation_helper(
                    np.concatenate((X_train, X_test)),
                    np.concatenate((T_train, T_test)),
                    np.concatenate((T_train_pre, T_test_pre))
                )  # (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                i_model_balance_over_kfold.append(result_all[5])
                i_model_fit_over_kfold.append(result_test[1])
                i_model_loss_over_kfold.append(result_test[0])

                self.results.append((i, k, para_d) + result_train + result_test + result_all)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]
            i_model_loss = [np.mean(i_model_loss_over_kfold), np.std(i_model_loss_over_kfold)]

            if self.criteria == "balance":
                if (i_model_balance[0] < self.best_balance) or \
                        ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
                    self.best_hyper_paras = para_d
                    self.best_balance = i_model_balance[0]
                    self.best_val = i_model_fit[0]
                    self.best_loss = i_model_loss[0]
                    self.best_balance_k_folds_detail = i_model_balance_over_kfold
                    self.best_val_k_folds_detail = i_model_fit_over_kfold
                    self.best_loss_k_folds_detail = i_model_loss_over_kfold
            elif self.criteria == "auroc":
                if i_model_fit[0] > self.best_val:
                    self.best_hyper_paras = para_d
                    self.best_balance = i_model_balance[0]
                    self.best_val = i_model_fit[0]
                    self.best_loss = i_model_loss[0]
                    self.best_balance_k_folds_detail = i_model_balance_over_kfold
                    self.best_val_k_folds_detail = i_model_fit_over_kfold
                    self.best_loss_k_folds_detail = i_model_loss_over_kfold
            elif self.criteria == "loss":
                if i_model_loss[0] < self.best_loss:
                    self.best_hyper_paras = para_d
                    self.best_balance = i_model_balance[0]
                    self.best_val = i_model_fit[0]
                    self.best_loss = i_model_loss[0]
                    self.best_balance_k_folds_detail = i_model_balance_over_kfold
                    self.best_val_k_folds_detail = i_model_fit_over_kfold
                    self.best_loss_k_folds_detail = i_model_loss_over_kfold
            else:
                raise NotImplementedError

            if i_model_fit[0] > self.global_best_val:
                self.global_best_val = i_model_fit[0]

            if i_model_balance[0] < self.global_best_balance:
                self.global_best_balance = i_model_balance[0]

            if i_model_loss[0] < self.global_best_loss:
                self.global_best_loss = i_model_loss[0]

            # save re-trained results on the whole data, for model selection exp only. Not necessary for later use
            model_retrain = self._model_estimation(para_d, X, T)
            T_pre = model_retrain.predict_proba_np(X)
            result_retrain = self._evaluation_helper(X, T, T_pre)
            self.results_retrain.append((i, 'retrain', para_d) + result_retrain)

            if verbose:
                if self.print_result:
                    self.report_stats()

        # end of training
        if self.print_result:
            print('best model parameter:', self.best_hyper_paras)
            print('re-training best model on all the data using best model parameter...')
        self.best_model = self._model_estimation(self.best_hyper_paras, X, T)
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain all_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        if self.print_result:
            print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def report_stats(self):
        print('Searching Space N={}: '.format(len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best balance: ', self.best_balance, ' Global Best balance: ', self.global_best_balance)
        print('Best fit value ', self.best_val, ' Global Best fit balue: ', self.global_best_val)
        try:
            pd.set_option('display.max_columns', None)
            describe = self.results.describe()
            print('AUC stats:\n', describe)
            return describe
        except:
            print('')

    def evaluate_test(self, X_test, T_test, X_test_all):
        # propensity scores on training and testing datasets
        T_test_pre = self.best_model.predict_proba_np(X_test)

        # evaluating goodness-of-balance and goodness-of-fit
        result_test = self._evaluation_helper(X_test_all, T_test, T_test_pre)

        return result_test

def cal_survival_HR_simple(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)
    if len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    # kmf = KaplanMeierFitter()
    T = golds_outcome[:, 1]
    treated_t2e, controlled_t2e = T[ones_idx], T[zeros_idx]
    # cox for hazard ratio
    cph = CoxPHFitter()
    event = golds_outcome[:, 0]
    event[event == -1] = 0
    weight = np.zeros(len(golds_treatment))
    weight[ones_idx] = treated_w.squeeze()
    weight[zeros_idx] = controlled_w.squeeze()
    cox_data = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment, 'weights': weight})
    try:
        cph.fit(cox_data, 'T', 'event', weights_col='weights', robust=True)
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
    except:
        cph = HR = CI = None
        cph_ori = HR_ori = CI_ori = None

    return (HR_ori, CI_ori, cph_ori), (HR, CI, cph)

def model_eval_common_simple(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5):
    y_pred_prob = logits_to_probability(PS_logits, normalized)
    # 1. IPTW sample weights
    treated_w, controlled_w = cal_weights(T, PS_logits, normalized=normalized, stabilized=True)
    treated_PS, control_PS = y_pred_prob[T == 1], y_pred_prob[T == 0]
    n_treat, n_control = (T == 1).sum(), (T == 0).sum()

    cox_HR_ori, cox_HR = cal_survival_HR_simple(T, PS_logits, Y, normalized)
    KM_ALL = (np.nan, np.nan, cox_HR_ori, cox_HR)

    if verbose:
        print('loss: {}'.format(loss))
        print('treated_weights:',
              pd.Series(treated_w.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_w))
        print('controlled_weights:', pd.Series(controlled_w.flatten()).describe().to_string().replace('\n',
                                                                                                      ';'))  # stats.describe(controlled_w))
        print('treated_PS:',
              pd.Series(treated_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_PS))
        print('controlled_PS:',
              pd.Series(control_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(control_PS))
        print('Cox Hazard ratio ori {} (CI: {})'.format(cox_HR_ori[0], cox_HR_ori[1]))
        print('Cox Hazard ratio iptw {} (CI: {})'.format(cox_HR[0], cox_HR[1]))

    return KM_ALL

def final_eval_ml_CV_revise(model, x, t, y):
    # ----. Model Evaluation & Final ATE results
    # model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5)

    print("*****" * 5, 'Evaluation on ALL data:')

    KM_ALL = model_eval_common_simple(X=x, T=t, Y=y, PS_logits=model.predict_proba(x)[:, 1])

    return KM_ALL
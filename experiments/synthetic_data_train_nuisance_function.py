import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append('./src')
import pickle
import numpy as np
import os
import os.path as osp
from ps import PropensityEstimator, cal_weights_align
from sklearn.model_selection import KFold
from eval_utils import evaluate_result_ContBinary, evaluate_result_ContBinary_DR, evaluate_covariate_balance
from causalml.inference.tf import DragonNet

import tensorflow as tf
import random
import torch


hidden_size_list = [50, 100, 200]
result_dir = "results"
N_SEED = 100

for GAMMA in [0.0, 5.0]:
    for seed in range(1, N_SEED + 1):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # load data
        with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}.pkl', 'rb') as f:
            train_a, train_y, train_Y1, train_Y0, train_X, train_X_raw, \
            test_a, test_y, test_Y1, test_Y0, test_X, test_X_raw = pickle.load(f)

            ps_model = PropensityEstimator(learner="LR", 
                                                paras_grid = {'C': 10 ** np.arange(-3, 3, 0.5)},
                                                criteria="balance",
                                                print_result=False)
            ps_model.cross_validation_fit(train_X, train_a)
            
            ps_test = ps_model.best_model.predict_proba(test_X)[:,1]
            _, _, ipw_test = cal_weights_align(test_a, ps_test, normalized=True,stabilized=False,clip=False)
            ipw_test = ipw_test[:, 0]
            
            ps_train = ps_model.best_model.predict_proba(train_X)[:,1]
            _, _, ipw_train = cal_weights_align(train_a, ps_train, normalized=True,stabilized=False,clip=False)
            ipw_train = ipw_train[:, 0]

        with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}-ipw.pkl', 'wb') as f:
            pickle.dump((ps_train, ipw_train, ps_test, ipw_test), f)

        cv_mse = []
        for hidden_size in hidden_size_list:
            kf = KFold(n_splits = 5, random_state=None, shuffle=False)
            val_mse = []
            for i, (tr_index, val_index) in enumerate(kf.split(train_X)):
                tr_X, val_X = train_X[tr_index], train_X[val_index]
                tr_a, val_a = train_a[tr_index], train_a[val_index]
                tr_y, val_y = train_y[tr_index], train_y[val_index]
            
                dragon = DragonNet(neurons_per_layer=hidden_size, epochs = 30, targeted_reg=True, verbose = False) # Dragonnet use 0.2 val_split by
                dragon.fit(train_X, train_a, train_y)
        
                val_pred_ys = dragon.predict(val_X)
                val_pred_y = val_pred_ys[:, 1] * val_a + val_pred_ys[:, 0] * (1 - val_a)
                val_mse.append(((val_pred_y - val_y) ** 2).mean())
        
            cv_mse.append(np.array(val_mse).mean()) 
        
        best_idx = np.array(cv_mse).argmin()
        best_hidden_size = hidden_size_list[best_idx]
        print(cv_mse, best_hidden_size)
        
        dragon = DragonNet(neurons_per_layer=best_hidden_size, epochs = 30, targeted_reg=True, verbose = False) # Dragonnet use 0.2 val_split by
        dragon.fit(train_X, train_a, train_y)
        
        pred_dragon_test = dragon.predict_tau(test_X).flatten()
        pred_dragon_train = dragon.predict_tau(train_X).flatten()
        
        pred_values_dragon_test = dragon.predict(test_X[:,:])
        pred_values_dragon_train = dragon.predict(train_X[:,:])
        
        pred_Y0_train, pred_Y1_train = pred_values_dragon_train[:, 0], pred_values_dragon_train[:, 1]
        pred_Y0_test, pred_Y1_test = pred_values_dragon_test[:, 0], pred_values_dragon_test[:, 1]
            
        items = []
        for expect_group_size in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            pred_dragon_threshold = np.quantile(pred_dragon_train, 1-expect_group_size)
            gt_cate = test_Y1[pred_dragon_test > pred_dragon_threshold].mean() - test_Y0[pred_dragon_test > pred_dragon_threshold].mean()
            iptw_cate_hard = evaluate_result_ContBinary(test_a[pred_dragon_test > pred_dragon_threshold], test_y[pred_dragon_test > pred_dragon_threshold], ipw_test[pred_dragon_test > pred_dragon_threshold])
            ate_aiptw = evaluate_result_ContBinary_DR(test_a[pred_dragon_test > pred_dragon_threshold], test_y[pred_dragon_test > pred_dragon_threshold], pred_Y1_test[pred_dragon_test > pred_dragon_threshold], pred_Y0_test[pred_dragon_test > pred_dragon_threshold], ipw_test[pred_dragon_test > pred_dragon_threshold])
            ate_cate = (pred_Y1_test[pred_dragon_test > pred_dragon_threshold]).mean() - (pred_Y0_test[pred_dragon_test > pred_dragon_threshold]).mean()
            n_unbalance, _, _ = evaluate_covariate_balance(test_X[pred_dragon_test > pred_dragon_threshold,:], test_a[pred_dragon_test > pred_dragon_threshold], 
                            ipw_test[pred_dragon_test > pred_dragon_threshold], device="cuda",smd_threshold = 0.2)
            # print(gt_cate, n_unbalance)
            g_size = (pred_dragon_test > pred_dragon_threshold).mean()
            items.append((g_size, gt_cate, iptw_cate_hard, ate_cate, ate_aiptw, n_unbalance))
        for i1, i2, i3, i4, i5, i6 in items:
            print(i1, i2, i3, i4, i5, i6)
        
        results_dragon = np.array(items)
        
        np.save(osp.join(result_dir, "GAMMA_{}".format(int(GAMMA)), "dragon_{}".format(seed)), results_dragon)
        with open(f'data/syn-gamma-{int(GAMMA)}/seed{seed}-dragonnet-output.pkl', 'wb') as f:
            pickle.dump((best_hidden_size, pred_Y1_train, pred_Y0_train, pred_Y1_test, pred_Y0_test), f)
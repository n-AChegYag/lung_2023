import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

def del_spaces(x):
    if type(x) is str:
        return x.strip()
    else:
        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The parameters of the training')
    parser.add_argument("-s", "--seed", type=int, required=False, default=26)
    args = parser.parse_args()
    
    # parameters
    excel_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/info_1012.xlsx'
    split_path = '/home/acy/data/lung/src/PL/splits/splits_231012_mc_c_16.pkl'
    random_seed = args.seed
    
    # splits
    with open(split_path) as f:
        splits = json.load(f)
        train_paths = splits['train']
        train_ids = []
        for path_to_patient in train_paths:
            patient_id = path_to_patient.split('/')[-1]
            train_ids.append(patient_id)
        valid_paths = splits['valid']
        valid_ids = []
        for path_to_patient in valid_paths:
            patient_id = path_to_patient.split('/')[-1]
            valid_ids.append(patient_id)
        test_paths  = splits['test']
        test_ids = []
        for path_to_patient in test_paths:
            patient_id = path_to_patient.split('/')[-1]
            test_ids.append(patient_id)
    test_1_ids = []
    test_2_ids = []
    for id in test_ids:
        if id[0:2] == 'js':
            test_2_ids.append(id)
        else:
            test_1_ids.append(id)
        
    # preprocessing data
    patients_info = pd.read_excel(excel_path)
    patients_info = pd.get_dummies(patients_info, columns=['性别'], drop_first=True)
    patients_info = pd.get_dummies(patients_info, columns=['TNM分期'])
    patients_info = pd.get_dummies(patients_info, columns=['临床分期'])
    patients_info = pd.get_dummies(patients_info, columns=['肿瘤位置'], drop_first=True)
    patients_info['patient_id'] = patients_info['patient_id'].astype(str)
    patients_info.set_index('patient_id', inplace=True)
    
    all_columns = list(patients_info)
    patients_info_train = pd.DataFrame(index=train_ids, columns=all_columns)
    for id in train_ids:
        patients_info_train.loc[id] = patients_info.loc[id]
    train_y = patients_info_train.iloc[:, 0].values.astype(int)
    train_r = patients_info_train.iloc[:, 1:11].values
    train_c = patients_info_train.iloc[:, 11:].values
    train_cr = patients_info_train.iloc[:, 1:].values
    scaler_r = MinMaxScaler().fit(train_r)
    scaler_c = MinMaxScaler().fit(train_c)
    scaler_cr = MinMaxScaler().fit(train_cr)
    train_r = scaler_r.transform(patients_info_train.iloc[:, 1:11].values)
    train_c = scaler_c.transform(patients_info_train.iloc[:, 11:].values)
    train_cr = scaler_cr.transform(patients_info_train.iloc[:, 1:].values)

    patients_info_valid = pd.DataFrame(index=valid_ids, columns=all_columns)
    for id in valid_ids:
        patients_info_valid.loc[id] = patients_info.loc[id]
    valid_y = patients_info_valid.iloc[:, 0].values.astype(int)
    valid_r = scaler_r.transform(patients_info_valid.iloc[:, 1:11].values)
    valid_c = scaler_c.transform(patients_info_valid.iloc[:, 11:].values)
    valid_cr = scaler_cr.transform(patients_info_valid.iloc[:, 1:].values)

    patients_info_test_1 = pd.DataFrame(index=test_1_ids, columns=all_columns)
    for id in test_1_ids:
        patients_info_test_1.loc[id] = patients_info.loc[id]
    test_1_y = patients_info_test_1.iloc[:, 0].values.astype(int)
    test_1_r = scaler_r.transform(patients_info_test_1.iloc[:, 1:11].values)
    test_1_c = scaler_c.transform(patients_info_test_1.iloc[:, 11:].values)
    test_1_cr = scaler_cr.transform(patients_info_test_1.iloc[:, 1:].values)
    
    patients_info_test_2 = pd.DataFrame(index=test_2_ids, columns=all_columns)
    for id in test_2_ids:
        patients_info_test_2.loc[id] = patients_info.loc[id]
    test_2_y = patients_info_test_2.iloc[:, 0].values.astype(int)
    test_2_r = scaler_r.transform(patients_info_test_2.iloc[:, 1:11].values)
    test_2_c = scaler_c.transform(patients_info_test_2.iloc[:, 11:].values)
    test_2_cr = scaler_cr.transform(patients_info_test_2.iloc[:, 1:].values)
    
    # define space
    space = [
        Integer(1, 100, name='n_estimators'),
        Integer(1, 100, name='max_depth'),
        Real(0.001, 1, name='min_samples_split'),
    ]
    rfc_temp = RandomForestClassifier(random_state=random_seed)
    
    # find opt params
    @use_named_args(space)
    def objective_c(**params):
        rfc_temp.set_params(**params)
        return - rfc_temp.fit(train_c, train_y).score(valid_c, valid_y)
    res_c = gp_minimize(objective_c, space, n_calls=100, random_state=random_seed)

    # modeling
    rfc_c = RandomForestClassifier(random_state=random_seed, n_estimators=res_c.x[0], max_depth=res_c.x[1], min_samples_split=res_c.x[2], n_jobs=-1)
    
    # fit c
    rfc_c.fit(train_c, train_y)
    prob_valid_c = rfc_c.predict_proba(valid_c)
    pred_valid_c = rfc_c.predict(valid_c)
    prob_test_1_c = rfc_c.predict_proba(test_1_c)
    prob_test_2_c = rfc_c.predict_proba(test_2_c)
    pred_test_1_c = rfc_c.predict(test_1_c)
    pred_test_2_c = rfc_c.predict(test_2_c)
    acc_valid_c = metrics.accuracy_score(valid_y, pred_valid_c)
    auc_valid_c = metrics.roc_auc_score(valid_y, prob_valid_c[:, 1])
    acc_test_1_c = metrics.accuracy_score(test_1_y, pred_test_1_c)
    auc_test_1_c = metrics.roc_auc_score(test_1_y, prob_test_1_c[:, 1])
    recall_test_1_c = metrics.recall_score(test_1_y, pred_test_1_c)
    precision_test_1_c = metrics.precision_score(test_1_y, pred_test_1_c)
    f1_test_1_c = metrics.f1_score(test_1_y, pred_test_1_c)
    acc_test_2_c = metrics.accuracy_score(test_2_y, pred_test_2_c)
    auc_test_2_c = metrics.roc_auc_score(test_2_y, prob_test_2_c[:, 1])
    recall_test_2_c = metrics.recall_score(test_2_y, pred_test_2_c)
    precision_test_2_c = metrics.precision_score(test_2_y, pred_test_2_c)
    f1_test_2_c = metrics.f1_score(test_2_y, pred_test_2_c)
    
    
    # find opt params
    @use_named_args(space)
    def objective_r(**params):
        rfc_temp.set_params(**params)
        return - rfc_temp.fit(train_r, train_y).score(valid_r, valid_y)
    res_r = gp_minimize(objective_r, space, n_calls=100, random_state=random_seed)

    # modeling
    rfc_r = RandomForestClassifier(random_state=random_seed, n_estimators=res_r.x[0], max_depth=res_r.x[1], min_samples_split=res_r.x[2], n_jobs=-1)
    
    # fit r
    rfc_r.fit(train_r, train_y)
    prob_valid_r = rfc_r.predict_proba(valid_r)
    pred_valid_r = rfc_r.predict(valid_r)
    prob_test_1_r = rfc_r.predict_proba(test_1_r)
    prob_test_2_r = rfc_r.predict_proba(test_2_r)
    pred_test_1_r = rfc_r.predict(test_1_r)
    pred_test_2_r = rfc_r.predict(test_2_r)
    acc_valid_r = metrics.accuracy_score(valid_y, pred_valid_r)
    auc_valid_r = metrics.roc_auc_score(valid_y, prob_valid_r[:, 1])
    acc_test_1_r = metrics.accuracy_score(test_1_y, pred_test_1_r)
    auc_test_1_r = metrics.roc_auc_score(test_1_y, prob_test_1_r[:, 1])
    recall_test_1_r = metrics.recall_score(test_1_y, pred_test_1_r)
    precision_test_1_r = metrics.precision_score(test_1_y, pred_test_1_r)
    f1_test_1_r = metrics.f1_score(test_1_y, pred_test_1_r)
    acc_test_2_r = metrics.accuracy_score(test_2_y, pred_test_2_r)
    auc_test_2_r = metrics.roc_auc_score(test_2_y, prob_test_2_r[:, 1])
    recall_test_2_r = metrics.recall_score(test_2_y, pred_test_2_r)
    precision_test_2_r = metrics.precision_score(test_2_y, pred_test_2_r)
    f1_test_2_r = metrics.f1_score(test_2_y, pred_test_2_r)

    # find opt params
    @use_named_args(space)
    def objective_cr(**params):
        rfc_temp.set_params(**params)
        return - rfc_temp.fit(train_cr, train_y).score(valid_cr, valid_y)
    res_cr = gp_minimize(objective_cr, space, n_calls=100, random_state=random_seed)

    # modeling
    rfc_cr = RandomForestClassifier(random_state=random_seed, n_estimators=res_cr.x[0], max_depth=res_cr.x[1], min_samples_split=res_cr.x[2], n_jobs=-1)
    
    # fit cr
    rfc_cr.fit(train_cr, train_y)
    prob_valid_cr = rfc_cr.predict_proba(valid_cr)
    pred_valid_cr = rfc_cr.predict(valid_cr)
    prob_test_1_cr = rfc_cr.predict_proba(test_1_cr)
    prob_test_2_cr = rfc_cr.predict_proba(test_2_cr)
    pred_test_1_cr = rfc_cr.predict(test_1_cr)
    pred_test_2_cr = rfc_cr.predict(test_2_cr)
    acc_valid_cr = metrics.accuracy_score(valid_y, pred_valid_cr)
    auc_valid_cr = metrics.roc_auc_score(valid_y, prob_valid_cr[:, 1])
    acc_test_1_cr = metrics.accuracy_score(test_1_y, pred_test_1_cr)
    auc_test_1_cr = metrics.roc_auc_score(test_1_y, prob_test_1_cr[:, 1])
    recall_test_1_cr = metrics.recall_score(test_1_y, pred_test_1_cr)
    precision_test_1_cr = metrics.precision_score(test_1_y, pred_test_1_cr)
    f1_test_1_cr = metrics.f1_score(test_1_y, pred_test_1_cr)
    acc_test_2_cr = metrics.accuracy_score(test_2_y, pred_test_2_cr)
    auc_test_2_cr = metrics.roc_auc_score(test_2_y, prob_test_2_cr[:, 1])
    recall_test_2_cr = metrics.recall_score(test_2_y, pred_test_2_cr)
    precision_test_2_cr = metrics.precision_score(test_2_y, pred_test_2_cr)
    f1_test_2_cr = metrics.f1_score(test_2_y, pred_test_2_cr)

    # report
    print('='*15)
    print(random_seed)
    print('acc')
    print(acc_test_1_c, acc_test_1_r, acc_test_1_cr)
    print(acc_test_2_c, acc_test_2_r, acc_test_2_cr)
    print('auc')
    print(auc_test_1_c, auc_test_1_r, auc_test_1_cr)
    print(auc_test_2_c, auc_test_2_r, auc_test_2_cr)
    print('recall')
    print(recall_test_1_c, recall_test_1_r, recall_test_1_cr)
    print(recall_test_2_c, recall_test_2_r, recall_test_2_cr)
    print('precision')
    print(precision_test_1_c, precision_test_1_r, precision_test_1_cr)
    print(precision_test_2_c, precision_test_2_r, precision_test_2_cr)
    print('f1')
    print(f1_test_1_c, f1_test_1_r, f1_test_1_cr)
    print(f1_test_2_c, f1_test_2_r, f1_test_2_cr) 
    print('='*15)
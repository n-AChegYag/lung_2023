
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def check_class_pred(patient, tag):
    if tag == 'CL':
        class_pred = int(patient.split('_')[1][0])
        if class_pred == 1:
            return 1
        elif class_pred == 0:
            return 0
    if tag == 'XT':
        class_pred = int(patient.split('_')[1][0])
        if class_pred == 0:
            return 1
        elif class_pred == 1:
            return 0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, required=False, default=16)
    parser.add_argument("-f", "--feats", type=str, required=False, default='c')
    parser.add_argument("-t", "--tag", type=str, required=False, default='231107_pre_cls_fake_mc4')
    parser.add_argument("-a", "--alpha", type=float, required=False, default=0.5)
    parser.add_argument("-l", "--lr", type=float, required=False, default=0.0001)
    args = parser.parse_args()
    
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical'
    path_to_pred = f'/home/acy/data/lung/log/{args.tag}_{args.seed}_{args.feats}_32_{args.lr}_{args.alpha}/lightning_logs/version_0'
    pred_paths = []
    for folder in os.listdir(path_to_pred):
        if folder.startswith('pred'):
            pred_paths.append(os.path.join(path_to_pred, folder))
    
    tags_list = ['cl', 'xt']
    
    for pred_path in pred_paths:
        pred_part = pred_path.split('/')[-1].split('_')[1]
    
        print('='*15)
        print(pred_path.split('/')[-1].split('_')[2])
        print(f'{args.tag}_{args.seed}_{args.feats}_{args.lr}_{args.alpha}')
        print(f'---{pred_part}---')
        
        if pred_part == 'test':
            prob_test_list, target_test_list = [], []
            prob_js_list, target_js_list = [], []
            patient_test_flag = 0
            patient_js_flag = 0
            tp_test, tn_test, fp_test, fn_test = 0, 0, 0, 0
            tp_js, tn_js, fp_js, fn_js = 0, 0, 0, 0
            for patient_ in tqdm(os.listdir(pred_path)):
                patient = patient_[:-2]
                pred = patient_[-1:]
                if patient[0:2] == 'js':
                    patient_js_flag += 1
                else:
                    patient_test_flag += 1
                with open(os.path.join(pred_path, patient_, 'prob.pth'), 'rb') as f:
                    prob = pickle.load(f)
                if patient[0:2] == 'js':
                    prob_js_list.append(prob[0,1])
                else:
                    prob_test_list.append(prob[0,1])
                if patient in os.listdir(os.path.join(data_path, 'cl')):
                    target = 1
                elif patient in os.listdir(os.path.join(data_path, 'xt')):
                    target = 0
                if patient[0:2] == 'js':
                    target_js_list.append(target)
                else:
                    target_test_list.append(target)
                if target == 1 and pred == '1':
                    if patient[0:2] == 'js':
                        tp_js += 1
                    else:
                        tp_test += 1
                elif target == 0 and pred == '0':
                    if patient[0:2] == 'js':
                        tn_js += 1
                    else:
                        tn_test += 1
                elif target == 1 and pred == '0':
                    if patient[0:2] == 'js':
                        fn_js += 1
                    else:
                        fn_test += 1
                elif target == 0 and pred == '1':
                    if patient[0:2] == 'js':
                        fp_js += 1
                    else:
                        fp_test += 1
            acc_test = (tp_test+tn_test)/patient_test_flag
            recall_test = tp_test/(tp_test+fn_test)
            precision_test = tp_test/(tp_test+fp_test)
            f1_test = 2*tp_test/(2*tp_test+fp_test+fn_test)
            auc_test = roc_auc_score(np.array(target_test_list), np.array(prob_test_list))
            acc_js = (tp_js+tn_js)/patient_js_flag
            recall_js = tp_js/(tp_js+fn_js)
            precision_js = tp_js/(tp_js+fp_js)
            f1_js = 2*tp_js/(2*tp_js+fp_js+fn_js)
            auc_js = roc_auc_score(np.array(target_js_list), np.array(prob_js_list))
            print('-'*15)
            print(f'test_acc : {acc_test:.4f}')
            print(f'test_recall : {recall_test:.4f}')
            print(f'test_precision : {precision_test:.4f}')
            print(f'test_f1 : {f1_test:.4f}')
            print(f'test_auc : {auc_test:.4f}')
            print('-'*15)
            print(f'js_acc : {acc_js:.4f}')
            print(f'js_recall : {recall_js:.4f}')
            print(f'js_precision : {precision_js:.4f}')
            print(f'js_f1 : {f1_js:.4f}')
            print(f'js_auc : {auc_js:.4f}')
        else:
            prob_list, target_list = [], []
            patient_flag = 0
            tp, tn, fp, fn = 0, 0, 0, 0
            for patient_ in tqdm(os.listdir(pred_path)):
                patient_flag += 1
                patient = patient_[:-2]
                pred = patient_[-1:]
                with open(os.path.join(pred_path, patient_, 'prob.pth'), 'rb') as f:
                    prob = pickle.load(f)
                prob_list.append(prob[0,1])
                if patient in os.listdir(os.path.join(data_path, 'cl')):
                    target = 1
                elif patient in os.listdir(os.path.join(data_path, 'xt')):
                    target = 0
                target_list.append(target)
                if target == 1 and pred == '1':
                    tp += 1
                elif target == 0 and pred == '0':
                    tn += 1
                elif target == 1 and pred == '0':
                    fn += 1
                elif target == 0 and pred == '1':
                    fp += 1
            acc = (tp+tn)/patient_flag
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*tp/(2*tp+fp+fn)
            auc = roc_auc_score(np.array(target_list), np.array(prob_list))
            print('-'*15)
            print(f'acc : {acc:.4f}')
            print(f'recall : {recall:.4f}')
            print(f'precision : {precision:.4f}')
            print(f'f1 : {f1:.4f}')
            print(f'auc : {auc:.4f}')
        print('='*15)
        print('\n')
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--part", type=str, required=False, default='real')
args = parser.parse_args()
    
def plot_roc_curve(results_dict, save_path):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.75)
    plt.plot(results_dict['c']['test1']['fpr'], results_dict['c']['test1']['tpr'], lw=2, label='Internal Validation Set, ROC curve (area = {:.4f})'.format(results_dict['c']['test1']['auc']), alpha=0.75)
    plt.plot(results_dict['c']['test2']['fpr'], results_dict['c']['test2']['tpr'], lw=2, label='External Validation Set, ROC curve (area = {:.4f})'.format(results_dict['c']['test2']['auc']), alpha=0.75)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic Curve of Validation Sets')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.75)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'roc_{args.part}_single.png'))
    
    
if __name__ == '__main__':
    
    feats_list = ['c', 'r', 'cr', 'cnn']
    tags_list = ['cl', 'xt']
    save_path = '/home/acy/data/lung_2023/results'
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical'
    results_dict = {}
    for feat in feats_list:
        results_dict[feat] = {}
        pred_path = f'/home/acy/data/lung_2023/logs/pre_cls_{feat}/pred_{args.part}/pred_test'
        prob_test_list, target_test_list = [], []
        prob_js_list, target_js_list = [], []
        patient_test_flag = 0
        patient_js_flag = 0
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

        threshold = 0.50
        pred_test_list = [1 if prob >= threshold else 0 for prob in prob_test_list]
        pred_js_list = [1 if prob >= threshold else 0 for prob in prob_js_list]
        fpr_test, tpr_test, thresholds_test = roc_curve(np.array(target_test_list), np.array(prob_test_list))
        fpr_js, tpr_js, thresholds_js = roc_curve(np.array(target_js_list), np.array(prob_js_list))
        auc_test = roc_auc_score(np.array(target_test_list), np.array(prob_test_list))
        auc_js = roc_auc_score(np.array(target_js_list), np.array(prob_js_list))
        results_dict[feat] = {
            'test1':    {
                'targets':  target_test_list,
                'probs':    prob_test_list,
                'preds':    pred_test_list,
                'fpr':      fpr_test,
                'tpr':      tpr_test,
                'auc':      auc_test,
            },
            'test2':    {
                'targets':  target_js_list,
                'probs':    prob_js_list,
                'preds':    pred_js_list,
                'fpr':      fpr_js,
                'tpr':      tpr_js,
                'auc':      auc_js,
            }
        }
        
    plot_roc_curve(results_dict, save_path)
    
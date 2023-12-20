import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Patch


if __name__ == '__main__':
    
    pred_path = '/home/acy/data/lung_2023/logs/pre_cls_c/pred_real/pred_test'
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical'
    save_path = '/home/acy/data/lung_2023/results'
    
    patients = [patient.split('_')[0] for patient in os.listdir(pred_path)]
    df_data = pd.DataFrame(index=patients, columns=['target', 'pred', 'prob', 'color'])
    for patient_ in tqdm(os.listdir(pred_path)):
        patient = patient_.split('_')[0]
        if patient in os.listdir(os.path.join(data_path, 'cl')):
            target = 1
        elif patient in os.listdir(os.path.join(data_path, 'xt')):
            target = 0
        color = 'red' if target == 1 else 'green'
        df_data.loc[patient, 'target'] = target
        df_data.loc[patient, 'color'] = color
        with open(os.path.join(pred_path, patient_, 'prob.pth'), 'rb') as f:
            prob = pickle.load(f)[0,1]
        if prob >= 0.5:
            pred = 1
        else:
            pred = 0
        df_data.loc[patient, 'pred'] = pred
        df_data.loc[patient, 'prob'] = prob
    df_data.sort_values('prob', inplace=True)
    
    categories = [
        'PR',
        'CR'
    ]
    
    # Draw plot
    plt.figure(figsize=(14,10), dpi=80)
    plt.hlines(y=df_data.index, xmin=0.5, xmax=df_data.prob, color=df_data.color, alpha=0.5, linewidth=5)
    
    # Decorations
    plt.gca().set(ylabel='$Patients$', xlabel='$score$')
    legend_element = [Patch(facecolor='red', label='PR', alpha=0.5), Patch(facecolor='green', label='CR', alpha=0.5)]
    plt.legend(handles=legend_element, loc='upper left')
    plt.yticks([])
    plt.title('Predicted Scores of All Patients from Two Test Sets', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_path, 'diverging_bar.png'))
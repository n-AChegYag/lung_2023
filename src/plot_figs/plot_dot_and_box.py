import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    
    pred_path = '/home/acy/data/lung_2023/logs/pre_cls_c/pred_real/pred_test'
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical'
    save_path = '/home/acy/data/lung_2023/results'
    
    patients = [patient.split('_')[0] for patient in os.listdir(pred_path)]
    df_data = pd.DataFrame(index=patients, columns=['target', 'pred', 'prob'])
    for patient_ in tqdm(os.listdir(pred_path)):
        patient = patient_.split('_')[0]
        if patient in os.listdir(os.path.join(data_path, 'cl')):
            target = 1
        elif patient in os.listdir(os.path.join(data_path, 'xt')):
            target = 0
        df_data.loc[patient, 'target'] = target
        with open(os.path.join(pred_path, patient_, 'prob.pth'), 'rb') as f:
            prob = pickle.load(f)[0,1]
        if prob >= 0.5:
            pred = 1
        else:
            pred = 0
        df_data.loc[patient, 'pred'] = pred
        df_data.loc[patient, 'prob'] = prob
    # df_data.reset_index(inplace=True)
    
    # Draw Plot
    plt.figure(figsize=(6,10), dpi= 80)
    sns.boxplot(x='target', y='prob', data=df_data, color='lightblue')
    sns.stripplot(x='target', y='prob', data=df_data, color='black', size=3)

    for i in range(len(df_data['target'].unique())-1):
        plt.vlines(i+0.5, 0, 1, linestyles='solid', colors='gray', alpha=0.2)

    # Decoration
    plt.title('Box Diagram')
    plt.savefig(os.path.join(save_path, 'box.png'))
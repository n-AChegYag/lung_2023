import os
import json 
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    
    src_path = '/home/acy/data/lung/log/inference_for_nnunet_b/output'
    pkl_path = '/home/acy/data/lung/src/PL/splits/splits_231012_mc_c_16.pkl'
    with open(pkl_path) as f:
        train_valid_split = json.load(f)
        test_patients = train_valid_split['test']
        valid_patients = train_valid_split['valid']
        train_patients = train_valid_split['train']
    patients = train_patients + valid_patients + test_patients
    for patient_path in tqdm(patients):
        patient_id = patient_path.split('/')[-1]
        shutil.copy(os.path.join(src_path, f'{patient_id}.nii.gz'), os.path.join(patient_path, 'pred.nii.gz'))
    
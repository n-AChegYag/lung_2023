import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage import measure

def get_largest_cd(tumor):
    spacing = tumor.GetSpacing()
    origin = tumor.GetOrigin()
    direction = tumor.GetDirection()
    tumor_array = sitk.GetArrayFromImage(tumor)
    labeled_tumor = measure.label(tumor_array)
    sizes = np.bincount(labeled_tumor.ravel())
    max_label = sizes[1:].argmax() + 1
    max_connected_area = (labeled_tumor == max_label)*1
    tumor = sitk.GetImageFromArray(max_connected_area)
    tumor.SetSpacing(spacing)
    tumor.SetOrigin(origin)
    tumor.SetDirection(direction)
    return tumor

if __name__ == '__main__':
    
    pkl_path = '/home/acy/data/lung/src/PL/splits/splits_231012_mc_c_16.pkl'
    with open(pkl_path) as f:
        train_valid_split = json.load(f)
        test_patients = train_valid_split['test']
        valid_patients = train_valid_split['valid']
        train_patients = train_valid_split['train']
    patients = train_patients + valid_patients + test_patients
    for patient in tqdm(patients):
        pred_tumor = sitk.ReadImage(os.path.join(patient, 'pred.nii.gz'))
        pred_tumor = get_largest_cd(pred_tumor)
        sitk.WriteImage(pred_tumor, os.path.join(patient, 'pred.nii.gz'))
        
    
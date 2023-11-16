import os
import SimpleITK as sitk
from tqdm import tqdm
from resampling import resample_data, check_and_create_path

if __name__ == '__main__':
    
    src_path = '/mnt/ssd2t/acy/lung_data/new_230615_nnunet_nii'
    des_path = '/home/acy/data/lung/data/clct_data'
    spacing = (1,1,5)
    
    for patient in tqdm(os.listdir(src_path)):
        check_and_create_path(os.path.join(des_path, patient))
        image = sitk.ReadImage(os.path.join(src_path, patient, 'image_a.nii.gz'))
        image = resample_data(image, spacing,'image')
        label = sitk.ReadImage(os.path.join(src_path, patient, 'label_a.nii.gz'))
        label = resample_data(label, spacing,'label')
        before = sitk.ReadImage(os.path.join(src_path, patient, 'label_b_warped.nii.gz'))
        before = resample_data(before, spacing,'label')
        sitk.WriteImage(image, os.path.join(des_path, patient, 'image.nii.gz'))
        sitk.WriteImage(label, os.path.join(des_path, patient, 'label.nii.gz'))
        sitk.WriteImage(before, os.path.join(des_path, patient, 'before.nii.gz'))
        
    
    
    
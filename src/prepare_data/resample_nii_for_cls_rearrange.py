import os
import resampling
import SimpleITK as sitk

from tqdm import tqdm

if __name__ == '__main__':
    
    nii_data_path = '/mnt/ssd2t/acy/lung_data_rearrange/temp'
    des_path = '/mnt/ssd2t/acy/lung_data_rearrange/temp_resample'
    tags = ['cl', 'xt']
    out_spacing = (1,1,5)

    for tag in tags:
        for patient in tqdm(os.listdir(os.path.join(nii_data_path, tag))):
            patient_path = os.path.join(nii_data_path, tag, patient)
            for nii_file in os.listdir(patient_path):
                resampling.check_and_create_path(os.path.join(des_path, tag, patient))
                if nii_file == 'image_before.nii.gz':
                    data = sitk.ReadImage(os.path.join(patient_path, nii_file))
                    resampled_data = resampling.resample_data(data, out_spacing, 'image')
                    sitk.WriteImage(resampled_data, os.path.join(des_path, tag, patient, nii_file))
                elif nii_file[0:7] == 'label_a':
                    data = sitk.ReadImage(os.path.join(patient_path, nii_file))
                    resampled_data = resampling.resample_data(data, out_spacing, 'label')
                    sitk.WriteImage(resampled_data, os.path.join(des_path, tag, patient, 'label_a.nii.gz'))
                elif nii_file[0:7] == 'label_b':
                    data = sitk.ReadImage(os.path.join(patient_path, nii_file))
                    resampled_data = resampling.resample_data(data, out_spacing, 'label')
                    sitk.WriteImage(resampled_data, os.path.join(des_path, tag, patient, 'label_b.nii.gz'))
                    
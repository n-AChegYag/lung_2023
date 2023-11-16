import os 
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    xt_path = '/mnt/ssd2t/acy/lung_data_rearrange/temp_resample/xt'
    # xt_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample/xt'

    for patient in tqdm(os.listdir(xt_path)):
        image = sitk.ReadImage(os.path.join(xt_path, patient, 'image_before.nii.gz'))
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        label_a = np.zeros_like(sitk.GetArrayFromImage(image))
        label_a = sitk.GetImageFromArray(label_a)
        label_a.SetSpacing(spacing)
        label_a.SetOrigin(origin)
        label_a.SetDirection(direction)
        sitk.WriteImage(label_a, os.path.join(xt_path, patient, 'label_a.nii.gz'))
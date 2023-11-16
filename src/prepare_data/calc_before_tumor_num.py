import os
import SimpleITK as sitk
from tqdm import tqdm
from skimage import measure

def count_con_doms(label_array):
    all_labels = measure.label(label_array)
    props = measure.regionprops(all_labels)
    return len(props)


if __name__ == '__main__':
    
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/temp_resample'
    tags = ['cl', 'xt']
    abs = ['a', 'b']
    
    for tag in tags:
        print(f'counting {tag} patients')
        for patient in tqdm(os.listdir(os.path.join(data_path, tag))):
            for ab in abs:
                if tag == 'xt' and ab == 'a':
                    continue
                tumor = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, tag, patient, f'label_{ab}.nii.gz')))
                tumor_num = count_con_doms(tumor)
                if tumor_num > 1:
                    print(f'{patient} | {ab} | {tumor_num}')
                
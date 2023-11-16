import os
import json
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    
    des_path = '/home/acy/data/lung/log/inference_for_nnunet_b/input'
    output_path = '/home/acy/data/lung/log/inference_for_nnunet_b/output'
    pkl_path = '/home/acy/data/lung/src/PL/splits/splits_231012_mc_c_16.pkl'
    shell_script = f'CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i {des_path} -o {output_path} -d 006 -c 3d_fullres -f all -chk checkpoint_best.pth --disable_tta'
    data_path = ''
    with open(pkl_path) as f:
        train_valid_split = json.load(f)
        test_patients = train_valid_split['test']
        valid_patients = train_valid_split['valid']
        train_patients = train_valid_split['train']
    patients = train_patients + valid_patients + test_patients
    # patients = test_patients
    patients = ['/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/xt/Lung251']
    map_dict = {}
    
    for idx, patient in enumerate(tqdm(patients)):
        idx = str(idx + 1).zfill(3)
        file_name = f'LungTumor_{idx}'
        map_dict[file_name] = patient.split('/')[-1]
        shutil.copyfile(os.path.join(patient, 'image_before.nii.gz'), os.path.join(des_path, file_name + '_0000.nii.gz'))
        
    os.system(shell_script)
    
    for k, v in tqdm(map_dict.items()):
        try:
            os.rename(os.path.join(output_path, f'{k}.nii.gz'), os.path.join(output_path, f'{v}.nii.gz'))
            # os.rename(os.path.join(output_path, f'{k}.npz'), os.path.join(output_path, f'{v}.npz'))
            os.rename(os.path.join(output_path, f'{k}.pkl'), os.path.join(output_path, f'{v}.pkl'))
        except:
            continue
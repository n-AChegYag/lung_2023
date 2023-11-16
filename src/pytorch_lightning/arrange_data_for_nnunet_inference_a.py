import os
import json
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    
    des_path = '/home/acy/data/lung/log/inference_for_nnunet_a/input'
    output_path = '/home/acy/data/lung/log/inference_for_nnunet_a/output'
    pkl_path = '/home/acy/data/lung/src/PL/splits/splits_231012_mc_c_16.pkl'
    shell_script = f'nnUNetv2_predict -i {des_path} -o {output_path} -d 007 -c 3d_fullres --save_probabilities -f all'
    with open(pkl_path) as f:
        train_valid_split = json.load(f)
        test_patients = train_valid_split['test']
    map_dict = {}
    
    for idx, patient in enumerate(tqdm(test_patients)):
        idx = str(idx + 1).zfill(3)
        file_name = f'LungTumor_{idx}'
        map_dict[file_name] = patient.split('/')[-1]
        shutil.copyfile(os.path.join(patient, 'image_before.nii.gz'), os.path.join(des_path, file_name + '_0000.nii.gz'))
        shutil.copyfile(os.path.join(patient, 'label_b.nii.gz'), os.path.join(des_path, file_name + '_0001.nii.gz'))
        
    os.system(shell_script)
    
    for k, v in tqdm(map_dict.items()):
        os.rename(os.path.join(output_path, f'{k}.nii.gz'), os.path.join(output_path, f'{v}.nii.gz'))
        os.rename(os.path.join(output_path, f'{k}.npz'), os.path.join(output_path, f'{v}.npz'))
        os.rename(os.path.join(output_path, f'{k}.pkl'), os.path.join(output_path, f'{v}.pkl'))
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage import measure
from sklearn.preprocessing import MinMaxScaler, scale


class LungDataset(Dataset):
    
    def __init__(self, data_list, mode='train', transforms=None, patch_size=(32,144,144), gt='pred'):
        super().__init__()
        # print('123')
        self.data_list = data_list
        self.mode = mode
        self.transforms = transforms
        self.patch_size = patch_size
        self.gt = gt
        self.all_images = [os.path.join(patient_id, 'image_before.nii.gz') for patient_id in self.data_list]
        if self.gt == 'manual':
            self.all_befores = [os.path.join(patient_id, 'label_b.nii.gz') for patient_id in self.data_list]
        elif self.gt == 'pred':
            self.all_befores = [os.path.join(patient_id, 'pred.nii.gz') for patient_id in self.data_list]
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            self.all_afters = [os.path.join(patient_id, 'label_a.nii.gz') for patient_id in self.data_list]
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        sample, info = {}, {}
        patient_id = self.data_list[index].split('/')[-2] + '/' + self.data_list[index].split('/')[-1]
        patient_id_ = self.data_list[index].split('/')[-1]
        info['patient_id'] = patient_id
        info['patient_id_'] = patient_id_
        image  = self._read_data(self.all_images[index])
        before = self._read_data(self.all_befores[index])
        info['ori_size'] = image.shape
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._read_data(self.all_afters[index])
            if patient_id[0:2] == 'cl':
                class_label = np.float32(1.)
            elif patient_id[0:2] == 'xt':
                class_label = np.float32(0.)
        # --- get patch --- #
        bbox = self._get_roi(before)
        info['bbox'] = bbox
        image  = self._get_patch(image, bbox)
        before = self._get_patch(before, bbox)
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._get_patch(after, bbox)
        # --- cat inputs --- #
        input_data = np.stack([image, before], axis=-1)
        sample['input'] = input_data
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = np.expand_dims(after, axis=3)
            assert input_data.shape[:-1] == after.shape[:-1], \
                f"Shape mismatch for the image with the shape {input_data.shape} and the mask with the shape {after.shape}."
            sample['target'] = after
            sample['class_label'] = class_label
        # --- augmentation --- #
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, info
    
    def _get_roi(self, tumor_array):
        out = [0,0,0,0,0,0]
        label = measure.label(tumor_array)
        prop = measure.regionprops(label)
        bbox = [999,999,999,0,0,0] # [min_z min_y min_x max_z max_y max_x]
        for i in prop:
            bbox[0] = i.bbox[0] if i.bbox[0] <= bbox[0] else bbox[0] 
            bbox[1] = i.bbox[1] if i.bbox[1] <= bbox[1] else bbox[1] 
            bbox[2] = i.bbox[2] if i.bbox[2] <= bbox[2] else bbox[2] 
            bbox[3] = i.bbox[3] if i.bbox[3] >= bbox[3] else bbox[3] 
            bbox[4] = i.bbox[4] if i.bbox[4] >= bbox[4] else bbox[4] 
            bbox[5] = i.bbox[5] if i.bbox[5] >= bbox[5] else bbox[5] 
        center = ((bbox[3]+bbox[0])/2, (bbox[4]+bbox[1])/2, (bbox[5]+bbox[2])/2) # [z, y, x]
        if center[0] - self.patch_size[0]/2 <= 0:
            out[0] = 0
            out[3] = 0 + self.patch_size[0]
        else:
            out[0] = center[0] - self.patch_size[0]/2
            out[3] = center[0] + self.patch_size[0]/2
        if center[1] - self.patch_size[1]/2 <= 0:
            out[1] = 0
            out[4] = 0 + self.patch_size[1]
        else:
            out[1] = center[1] - self.patch_size[1]/2
            out[4] = center[1] + self.patch_size[1]/2
        if center[2] - self.patch_size[2]/2 <= 0:
            out[2] = 0
            out[5] = 0 + self.patch_size[2]
        else:
            out[2] = center[2] - self.patch_size[2]/2
            out[5] = center[2] + self.patch_size[2]/2
        if center[0] + self.patch_size[0]/2 > tumor_array.shape[0]:
            out[0] = tumor_array.shape[0] - self.patch_size[0]
            out[3] = tumor_array.shape[0]
        if center[1] + self.patch_size[1]/2 > tumor_array.shape[1]:
            out[1] = tumor_array.shape[1] - self.patch_size[1]
            out[4] = tumor_array.shape[1]
        if center[2] + self.patch_size[2]/2 > tumor_array.shape[2]:
            out[2] = tumor_array.shape[2] - self.patch_size[2]
            out[5] = tumor_array.shape[2]
        for idx, i in enumerate(out):
            out[idx] = int(i)
        return out

    @staticmethod
    def _read_data(path, return_array=True):
        if return_array:
            return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype('float32')
        return sitk.ReadImage(path)
    
    @staticmethod
    def _get_patch(data, bbox):
        return data[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]   
    
    def get_labels(self):
        labels = []
        for index in range(len(self.data_list)):
            labels.append(self.__getitem__(index)[0]['class_label'])
        return labels
    

class LungRadiomicsDataset_(LungDataset):
    
    def __init__(self, data_list, radiomics_path, mode='train', transforms=None, patch_size=(32, 144, 144), gt='pred'):
        super().__init__(data_list, mode, transforms, patch_size, gt)
        self.radiomics_path = radiomics_path
        self.radiomics_feats = pd.read_excel(self.radiomics_path)
        self.radiomics_feats.set_index('patient_id', inplace=True)
        self.feats_name = self.radiomics_feats.columns.values.tolist()[1:]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        for f_n in self.feats_name:
            # self.radiomics_feats[f_n] = self.scaler.fit_transform(self.radiomics_feats[[f_n]])
            self.radiomics_feats[f_n] = scale(self.radiomics_feats[f_n])

    def __getitem__(self, index):
        sample, info = {}, {}
        patient_id = self.data_list[index].split('/')[-2] + '/' + self.data_list[index].split('/')[-1]
        patient_id_ = self.data_list[index].split('/')[-1]
        info['patient_id'] = patient_id
        info['patient_id_'] = patient_id_
        image  = self._read_data(self.all_images[index])
        before = self._read_data(self.all_befores[index])
        radiomics_feature = self.radiomics_feats.loc[patient_id_]
        sample['feature'] = pd.to_numeric(radiomics_feature[1:], errors='coerce').values.astype(np.float32)
        info['ori_size'] = image.shape
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._read_data(self.all_afters[index])
            if patient_id[0:2] == 'cl':
                class_label = np.float32(1.)
            elif patient_id[0:2] == 'xt':
                class_label = np.float32(0.)
        # --- get patch --- #
        bbox = self._get_roi(before)
        info['bbox'] = bbox
        image  = self._get_patch(image, bbox)
        before = self._get_patch(before, bbox)
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._get_patch(after, bbox)
        # --- cat inputs --- #
        input_data = np.stack([image, before], axis=-1)
        sample['input'] = input_data
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = np.expand_dims(after, axis=3)
            assert input_data.shape[:-1] == after.shape[:-1], \
                f"Shape mismatch for the image with the shape {input_data.shape} and the mask with the shape {after.shape}."
            sample['target'] = after
            sample['class_label'] = class_label
        # --- augmentation --- #
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, info
    
    
class LungRadiomicsClinicalDataset(LungDataset):
    
    def __init__(self, data_list, info_path, mode='train', transforms=None, patch_size=(32, 144, 144), gt='pred'):
        super().__init__(data_list, mode, transforms, patch_size, gt)
        self.info_path = info_path
        self.patients_info = pd.read_excel(self.info_path)
        self.patients_info = self.patients_info.applymap(self.del_spacea)
        self.patients_info['wavelet-LLL_ngtdm_Strength'] = scale(self.patients_info['wavelet-LLL_ngtdm_Strength'])
        self.patients_info['wavelet-LLL_gldm_LowGrayLevelEmphasis'] = scale(self.patients_info['wavelet-LLL_gldm_LowGrayLevelEmphasis'])
        self.patients_info['wavelet-HHH_glszm_LargeAreaHighGrayLevelEmphasis'] = scale(self.patients_info['wavelet-HHH_glszm_LargeAreaHighGrayLevelEmphasis'])
        self.patients_info['wavelet-HHH_glszm_GrayLevelNonUniformity'] = scale(self.patients_info['wavelet-HHH_glszm_GrayLevelNonUniformity'])
        self.patients_info['wavelet-HLH_glszm_LargeAreaHighGrayLevelEmphasis'] = scale(self.patients_info['wavelet-HLH_glszm_LargeAreaHighGrayLevelEmphasis'])
        self.patients_info['wavelet-HLH_gldm_SmallDependenceLowGrayLevelEmphasis'] = scale(self.patients_info['wavelet-HLH_gldm_SmallDependenceLowGrayLevelEmphasis'])
        self.patients_info['gradient_glszm_LowGrayLevelZoneEmphasis'] = scale(self.patients_info['gradient_glszm_LowGrayLevelZoneEmphasis'])
        self.patients_info['wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis'] = scale(self.patients_info['wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis'])
        self.patients_info['wavelet-HHH_firstorder_Kurtosis'] = scale(self.patients_info['wavelet-HHH_firstorder_Kurtosis'])
        self.patients_info['wavelet-HHH_glszm_SmallAreaLowGrayLevelEmphasis'] = scale(self.patients_info['wavelet-HHH_glszm_SmallAreaLowGrayLevelEmphasis'])
        self.patients_info['年龄'] = scale(self.patients_info['年龄'])
        self.patients_info['治疗剂量'] = scale(self.patients_info['治疗剂量'])
        self.patients_info['次数'] = scale(self.patients_info['次数'])
        self.patients_info['BED'] = scale(self.patients_info['BED'])
        self.patients_info = pd.get_dummies(self.patients_info, columns=['性别'], drop_first=True)
        self.patients_info = pd.get_dummies(self.patients_info, columns=['TNM分期'])
        self.patients_info = pd.get_dummies(self.patients_info, columns=['临床分期'])
        self.patients_info = pd.get_dummies(self.patients_info, columns=['肿瘤位置'], drop_first=True)
        self.patients_info['patient_id'] = self.patients_info['patient_id'].astype(str)
        self.patients_info.set_index('patient_id', inplace=True)
        
    def __getitem__(self, index):
        sample, info = {}, {}
        patient_id = self.data_list[index].split('/')[-2] + '/' + self.data_list[index].split('/')[-1]
        patient_id_ = self.data_list[index].split('/')[-1]
        info['patient_id'] = patient_id
        info['patient_id_'] = patient_id_
        image  = self._read_data(self.all_images[index])
        before = self._read_data(self.all_befores[index])
        feature = self.patients_info.loc[patient_id_]
        sample['feature'] = pd.to_numeric(feature[1:], errors='coerce').values.astype(np.float32)
        info['ori_size'] = image.shape
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._read_data(self.all_afters[index])
            if patient_id[0:2] == 'cl':
                class_label = np.float32(1.)
            elif patient_id[0:2] == 'xt':
                class_label = np.float32(0.)
        # --- get patch --- #
        bbox = self._get_roi(before)
        info['bbox'] = bbox
        image  = self._get_patch(image, bbox)
        before = self._get_patch(before, bbox)
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._get_patch(after, bbox)
        # --- cat inputs --- #
        input_data = np.stack([image, before], axis=-1)
        sample['input'] = input_data
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = np.expand_dims(after, axis=3)
            assert input_data.shape[:-1] == after.shape[:-1], \
                f"Shape mismatch for the image with the shape {input_data.shape} and the mask with the shape {after.shape}."
            sample['target'] = after
            sample['class_label'] = class_label
        # --- augmentation --- #
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, info
    
    @staticmethod
    def del_spacea(x):
        if type(x) is str:
            return x.strip()
        else:
            return x
        
class LungClinicalDataset(LungRadiomicsClinicalDataset):
    
    def __init__(self, data_list, info_path, mode='train', transforms=None, patch_size=(32, 144, 144), gt='pred'):
        super().__init__(data_list, info_path, mode, transforms, patch_size, gt)
        
    def __getitem__(self, index):
        sample, info = {}, {}
        patient_id = self.data_list[index].split('/')[-2] + '/' + self.data_list[index].split('/')[-1]
        patient_id_ = self.data_list[index].split('/')[-1]
        info['patient_id'] = patient_id
        info['patient_id_'] = patient_id_
        image  = self._read_data(self.all_images[index])
        before = self._read_data(self.all_befores[index])
        feature = self.patients_info.loc[patient_id_]
        sample['feature'] = pd.to_numeric(feature[1:], errors='coerce').values.astype(np.float32)[10:]
        info['ori_size'] = image.shape
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._read_data(self.all_afters[index])
            if patient_id[0:2] == 'cl':
                class_label = np.float32(1.)
            elif patient_id[0:2] == 'xt':
                class_label = np.float32(0.)
        # --- get patch --- #
        bbox = self._get_roi(before)
        info['bbox'] = bbox
        image  = self._get_patch(image, bbox)
        before = self._get_patch(before, bbox)
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._get_patch(after, bbox)
        # --- cat inputs --- #
        input_data = np.stack([image, before], axis=-1)
        sample['input'] = input_data
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = np.expand_dims(after, axis=3)
            assert input_data.shape[:-1] == after.shape[:-1], \
                f"Shape mismatch for the image with the shape {input_data.shape} and the mask with the shape {after.shape}."
            sample['target'] = after
            sample['class_label'] = class_label
        # --- augmentation --- #
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, info
    
class LungRadiomicsDataset(LungRadiomicsClinicalDataset):
    
    def __init__(self, data_list, info_path, mode='train', transforms=None, patch_size=(32, 144, 144), gt='pred'):
        super().__init__(data_list, info_path, mode, transforms, patch_size, gt)

    def __getitem__(self, index):
        sample, info = {}, {}
        patient_id = self.data_list[index].split('/')[-2] + '/' + self.data_list[index].split('/')[-1]
        patient_id_ = self.data_list[index].split('/')[-1]
        info['patient_id'] = patient_id
        info['patient_id_'] = patient_id_
        image  = self._read_data(self.all_images[index])
        before = self._read_data(self.all_befores[index])
        feature = self.patients_info.loc[patient_id_]
        sample['feature'] = pd.to_numeric(feature[1:], errors='coerce').values.astype(np.float32)[0:10]
        info['ori_size'] = image.shape
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._read_data(self.all_afters[index])
            if patient_id[0:2] == 'cl':
                class_label = np.float32(1.)
            elif patient_id[0:2] == 'xt':
                class_label = np.float32(0.)
        # --- get patch --- #
        bbox = self._get_roi(before)
        info['bbox'] = bbox
        image  = self._get_patch(image, bbox)
        before = self._get_patch(before, bbox)
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = self._get_patch(after, bbox)
        # --- cat inputs --- #
        input_data = np.stack([image, before], axis=-1)
        sample['input'] = input_data
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            after = np.expand_dims(after, axis=3)
            assert input_data.shape[:-1] == after.shape[:-1], \
                f"Shape mismatch for the image with the shape {input_data.shape} and the mask with the shape {after.shape}."
            sample['target'] = after
            sample['class_label'] = class_label
        # --- augmentation --- #
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, info
    
    
class LungSegDataset(LungDataset):
    
    def __init__(self, data_list, mode='train', transforms=None, patch_size=(32, 144, 144), gt='pred'):
        super().__init__(data_list, mode, transforms, patch_size, gt)
        
    def __getitem__(self, index):
        sample, info = {}, {}
        patient_id = self.data_list[index].split('/')[-2] + '/' + self.data_list[index].split('/')[-1]
        patient_id_ = self.data_list[index].split('/')[-1]
        info['patient_id'] = patient_id
        info['patient_id_'] = patient_id_
        image  = self._read_data(self.all_images[index])
        info['ori_size'] = image.shape
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            before = self._read_data(self.all_befores[index])
        # --- get patch --- #
        bbox = self._get_roi(before)
        info['bbox'] = bbox
        image  = self._get_patch(image, bbox)
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            before = self._get_patch(before, bbox)
        # --- cat inputs --- #
        input_data = np.expand_dims(image, axis=3)
        sample['input'] = input_data
        if self.mode == 'train' or self.mode == 'valid' or self.mode == 'test':
            before = np.expand_dims(before, axis=3)
            assert input_data.shape[:-1] == before.shape[:-1], \
                f"Shape mismatch for the image with the shape {input_data.shape} and the mask with the shape {before.shape}."
            sample['target'] = before
        # --- augmentation --- #
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, info
            
        
        
if __name__ == '__main__':
    
    import json
    from torch.utils.data import DataLoader

    path_to_pkl = '/home/acy/data/lung/src/PL/splits/split_230827_c_0.pkl'
    path_to_info = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/info_0826.xlsx'
    
    with open(path_to_pkl) as f:
        train_valid_split = json.load(f)
        train_paths = train_valid_split['train']
    
    dataset = LungRadiomicsClinicalDataset(train_paths, path_to_info)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for sample, info in dataloader:
        try:
            print(sample['input'].shape)
            print(sample['feature'])
        except:
            print(info['patient_id'])
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor
from tqdm import tqdm
from skimage import measure

class RadiomicsFeatureExtractor():

    def __init__(self, data_path, yaml_path, save_path):
        self.data_path = data_path
        self.yaml_path = yaml_path
        self.save_path = save_path
        self.classes = ['cl', 'xt']
        self.patients = [os.path.join(data_path, CL_or_XT, patient) for CL_or_XT in self.classes for patient in os.listdir(os.path.join(data_path, CL_or_XT))]
        self.extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)
        self.extractor.addProvenance(False)

    def init_df(self):
        image, tumor = self._get_clipped_image_and_tumor(self.patients[0])
        feature_vector = self.extractor.execute(image, tumor)
        self.df_feats=pd.DataFrame(index=range(len(self.patients)), columns=feature_vector.keys())
        self.df_labels=pd.DataFrame(index=range(len(self.patients)), columns=['label', 'patient'])

    def extract_features(self):
        for idx, patient in enumerate(tqdm(self.patients)):
            try:
                patient_id = patient.split('/')[-1]
                self.df_feats.loc[idx, 'patient_id'] = patient_id
                self.df_labels.loc[idx, 'patient_id'] = patient_id
                image, tumor = self._get_clipped_image_and_tumor(patient)
                feature_vector = self.extractor.execute(image, tumor)
                for feature_name in feature_vector.keys():
                    self.df_feats.loc[idx, feature_name] = feature_vector[feature_name]
                self.df_labels.loc[idx, 'label'] = 1 if patient.split('/')[-2] == 'cl' else 0
                self.df_labels.loc[idx, 'patient'] = patient_id
            except:
                print('Something wrong... [{}]'.format(patient.split('/')[-1]))
        self.df_feats.to_excel(os.path.join(self.save_path, 'feats_1107.xlsx'), sheet_name='feats')
        self.df_labels.to_excel(os.path.join(self.save_path, 'labels_1107.xlsx'), sheet_name='labels')
    
    def _get_clipped_image_and_tumor(self, patient_path, free=3):
        image = sitk.ReadImage(os.path.join(patient_path, 'image_before.nii.gz'))
        # tumor = sitk.ReadImage(os.path.join(patient_path, 'label_b.nii.gz'))
        tumor = sitk.ReadImage(os.path.join(patient_path, 'pred.nii.gz'))
        tumor_array = sitk.GetArrayFromImage(tumor)
        tumor_array = self._get_largest_connected_domain(tumor_array)
        roi_list = np.array(list(set(tumor_array.nonzero()[0])), dtype=np.int32)
        roi_index = range(max(roi_list[0] - free, 0), min(roi_list[-1] + free, tumor_array.shape[0]))
        image, tumor = self._clip_image(image, tumor, roi_index)
        return image, tumor

    @staticmethod
    def _get_largest_connected_domain(tumor_array):
        labeled_tumor = measure.label(tumor_array)
        sizes = np.bincount(labeled_tumor.ravel())
        max_label = sizes[1:].argmax() + 1
        max_connected_area = (labeled_tumor == max_label)*1
        return max_connected_area

    @staticmethod
    def _clip_image(image, tumor, index):
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        image_array = sitk.GetArrayFromImage(image)
        image_clipped = sitk.GetImageFromArray(image_array[index,:,:])
        image_clipped.SetSpacing(spacing)
        image_clipped.SetOrigin(origin)
        image_clipped.SetDirection(direction)
        tumor_array = sitk.GetArrayFromImage(tumor)
        tumor_clipped = sitk.GetImageFromArray(tumor_array[index,:,:])
        tumor_clipped.SetSpacing(spacing)
        tumor_clipped.SetOrigin(origin)
        tumor_clipped.SetDirection(direction)
        return image_clipped, tumor_clipped




if __name__ == '__main__':
        
        DATA_PATH = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical'
        YAML_PATH = 'radiomics/acy.yaml'
        SAVE_PATH = 'radiomics'

        MyExtractor = RadiomicsFeatureExtractor(DATA_PATH, YAML_PATH, SAVE_PATH)
        MyExtractor.init_df()
        MyExtractor.extract_features()

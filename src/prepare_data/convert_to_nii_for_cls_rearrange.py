import os
from tqdm import tqdm
import SimpleITK as sitk
from DicomRTTool import DicomReaderWriter

def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_to_nii(dicom_path, save_path, tag='cl'):
    check_and_create_folder(os.path.join(save_path))
    reader = DicomReaderWriter()
    reader.walk_through_folders(dicom_path)
    reader.get_images()
    sitk.WriteImage(reader.dicom_handle, os.path.join(save_path, f'image_after.nii.gz'))
    reader.return_rois(print_rois=True)
    for roi in reader.rois_in_case:
        reader.set_contour_names_and_associations([roi])
        reader.get_mask()
        if tag == 'cl':
            if ('gtv' in roi) and ('after' not in roi):
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_b_{roi}.nii.gz'))
            if ('gtv' in roi) and ('after' in roi):
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_a_{roi}.nii.gz'))
        elif tag == 'xt':
            if 'gtv' in roi:
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_b_{roi}.nii.gz'))
        
if __name__ == "__main__":

    ori_data_path = '/mnt/ssd2t/acy/lung_data_rearrange/ori_data/all'
    des_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls'
    tags = ['cl', 'xt']
    
    for tag in tags:
        for patient in tqdm(os.listdir(os.path.join(ori_data_path, tag))): 
            if 'before' in os.listdir(os.path.join(ori_data_path, tag, patient)):
                dicom_path = os.path.join(ori_data_path, tag, patient, 'before')
            elif 'Before' in os.listdir(os.path.join(ori_data_path, tag, patient)):
                dicom_path = os.path.join(ori_data_path, tag, patient, 'Before')
            else:
                dicom_path = os.path.join(ori_data_path, tag, patient)
            try:
                convert_to_nii(dicom_path, os.path.join(des_path, tag, patient), tag)
            except AttributeError:
                continue
        
        
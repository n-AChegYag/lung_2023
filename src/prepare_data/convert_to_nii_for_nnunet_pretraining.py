import os
import SimpleITK as sitk
from DicomRTTool import DicomReaderWriter

DATA_PATH = '/mnt/ssd2t/acy/lung_data/Lung_ai20230505/canliu'
SAVE_PATH = '/mnt/ssd2t/acy/lung_data/new_230615_nnunet_nii'


def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_to_nii(raw_path, save_path, save_image=True, label_flag='bb'):
    check_and_create_folder(os.path.join(save_path))
    reader = DicomReaderWriter()
    reader.walk_through_folders(raw_path)
    reader.get_images()
    if save_image:
        sitk.WriteImage(reader.dicom_handle, os.path.join(save_path, f'image_{label_flag[0]}.nii.gz'))
    reader.return_rois(print_rois=True)
    for roi in reader.rois_in_case:
        reader.set_contour_names_and_associations([roi])
        reader.get_mask()
        # save before-ct's before-tumor
        if label_flag == 'bb':
            if 'gtv' in roi and 'after' not in roi:
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_b_{roi}.nii.gz'))
        # save after-ct's after-tumor
        elif label_flag == 'aa':
            if 'gtv' in roi:
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_a_{roi}.nii.gz'))
        # save before-ct's after-tumor
        elif label_flag == 'ba':
            if 'gtv' in roi and 'after' in roi:
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_a_{roi}.nii.gz'))
        # save after-ct's before-tumor
        elif label_flag == 'ab':
            if 'gtv' in roi:
                sitk.WriteImage(reader.annotation_handle, os.path.join(save_path, f'label_b_{roi}.nii.gz'))



if __name__ == '__main__':

    all_ins = os.listdir(DATA_PATH)
    for patient in all_ins:

        try:
            if os.path.exists(os.path.join(DATA_PATH, patient, 'After')):
                current_path = os.path.join(DATA_PATH, patient, 'After')
            elif os.path.exists(os.path.join(DATA_PATH, patient, 'after')):
                current_path = os.path.join(DATA_PATH, patient, 'after')
            else:
                print(f'something going wrong with {patient}')
                continue
            convert_to_nii(current_path, os.path.join(SAVE_PATH, patient), True, 'aa')

            if os.path.exists(os.path.join(DATA_PATH, patient, 'Before')):
                current_path = os.path.join(DATA_PATH, patient, 'Before')
            elif os.path.exists(os.path.join(DATA_PATH, patient, 'before')):
                current_path = os.path.join(DATA_PATH, patient, 'before')
            else:
                print(f'something going wrong with {patient}')
                continue
            convert_to_nii(current_path, os.path.join(SAVE_PATH, patient), True, 'bb')

        except AttributeError:
            continue
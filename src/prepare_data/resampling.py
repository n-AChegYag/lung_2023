import os
import SimpleITK as sitk


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resample_data(data, out_spacing, image_or_label):
    ori_size = data.GetSize()
    out_size = [0, 0, 0]
    ori_spacing = data.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()
    for i in range(3):
        out_size[i] = int(ori_size[i] * ori_spacing[i] / out_spacing[i] + 0.5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    if image_or_label == 'image':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif image_or_label == 'label':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(data.GetOrigin())
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputDirection(data.GetDirection())
    resampler.SetSize(out_size)
    output = resampler.Execute(data)
    return output

if __name__ == '__main__':

    from tqdm import tqdm
    
    DATA_PATH = '/mnt/ssd2t/acy/lung_data/new_230509_nii'
    SAVE_PATH = '/mnt/ssd2t/acy/lung_data/new_230509_115'
    check_and_create_path(SAVE_PATH)
    out_spacing = (1, 1, 5)

    for patient in tqdm(os.listdir(DATA_PATH)):
        check_and_create_path(os.path.join(SAVE_PATH, patient))
        for fname in os.listdir(os.path.join(DATA_PATH, patient)):
            if fname[0:5] == 'image':
                image = sitk.ReadImage(os.path.join(DATA_PATH, patient, fname))
                image = resample_data(image, out_spacing, 'image')
                sitk.WriteImage(image, os.path.join(SAVE_PATH, patient, fname))
            elif fname[0:5] == 'label':
                label = sitk.ReadImage(os.path.join(DATA_PATH, patient, fname))
                label = resample_data(label, out_spacing, 'label')
                sitk.WriteImage(label, os.path.join(SAVE_PATH, patient, fname))
                # sitk.WriteImage(label, os.path.join(SAVE_PATH, patient, 'label.nii.gz'))
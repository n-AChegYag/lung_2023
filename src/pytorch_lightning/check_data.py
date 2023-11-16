import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from skimage import measure
from scipy import spatial


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

def compute_tumor_diameter(tumor_111_array):
    # by Li Ruikun
    tumor_region = measure.label(tumor_111_array, background=0)
    properties = measure.regionprops(tumor_region)
    prop = properties[0]
    points = prop.coords
    candidates = points[spatial.ConvexHull(points).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    dis_i, dis_j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    dxx = abs(candidates[dis_i][0] - candidates[dis_j][0])
    dyy = abs(candidates[dis_i][1] - candidates[dis_j][1])
    dzz = abs(candidates[dis_i][2] - candidates[dis_j][2])
    diameter = (dxx ** 2 + dyy ** 2 + dzz ** 2) ** 0.5
    return diameter


if __name__ == '__main__':
    
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample'
    tags = ['cl', 'xt']
    for tag in tags:
        for patient in tqdm(os.listdir(os.path.join(data_path, tag))):
            try:
                tumor_before = sitk.ReadImage(os.path.join(data_path, tag, patient, 'label_b.nii.gz'))
                tumor_before = resample_data(tumor_before, (1,1,1), 'label')
                tumor_before_array = sitk.GetArrayFromImage(tumor_before)
                tumor_diameter = compute_tumor_diameter(tumor_before_array)
            except:
                print(os.path.join(data_path, tag, patient, 'label_b.nii.gz'))
    

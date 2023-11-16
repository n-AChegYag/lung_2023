import os
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import measure
from scipy import spatial
from resampling import resample_data

def compute_recall(tumor_before, tumor_after):
    tumor_before_array = sitk.GetArrayFromImage(tumor_before)
    tumor_after_array = sitk.GetArrayFromImage(tumor_after)
    fp = np.sum(np.logical_and(np.logical_not(tumor_after_array), tumor_before_array))
    tp = np.sum(np.logical_and(tumor_before_array, tumor_after_array))
    fn = np.sum(np.logical_and(np.logical_not(tumor_before_array), tumor_after_array))
    recall = tp / (tp + fn)
    return recall

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
    
    parser = argparse.ArgumentParser(description='The parameters')

    parser.add_argument("-t", "--threshold", type=float, required=False, default=0.90,
                        help="threshold of recall")
    args = parser.parse_args()
    
    cl_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/cl'
    # threshold = 0.80 # 1.00 or 0.90 or 0.80
    
    BE = sitk.BinaryErodeImageFilter()
    BE.SetKernelType(sitk.sitkBall)
    BE.SetKernelRadius(1)
    BE.SetForegroundValue(1)
    BE.SetBackgroundValue(0)
    
    erode_df = pd.DataFrame(index=os.listdir(cl_path), columns=['distance', 'volume_ratio_1', 'volume_ratio_2', 'diameter'])
    
    for patient in tqdm(os.listdir(cl_path)):
        tumor_before = sitk.ReadImage(os.path.join(cl_path, patient, 'label_b.nii.gz'))
        tumor_after = sitk.ReadImage(os.path.join(cl_path, patient, 'label_a.nii.gz'))
        tumor_before = resample_data(tumor_before, (1,1,1), 'label')
        tumor_after = resample_data(tumor_after, (1,1,1), 'label')
        d = compute_tumor_diameter(sitk.GetArrayFromImage(tumor_before))
        erode_df.loc[patient, 'diameter'] = d
        recall = compute_recall(tumor_before, tumor_after)
        if recall >= args.threshold:
            for erode_distance in range(1,20):
                BE.SetKernelRadius(erode_distance)
                tumor_before_erode = BE.Execute(tumor_before)
                recall = compute_recall(tumor_before_erode, tumor_after)
                if recall < args.threshold:
                    volume_eroded = np.sum(sitk.GetArrayFromImage(tumor_before_erode))
                    volume_before = np.sum(sitk.GetArrayFromImage(tumor_before))
                    volume_after = np.sum(sitk.GetArrayFromImage(tumor_after))
                    erode_df.loc[patient, 'distance'] = erode_distance
                    erode_df.loc[patient, 'volume_ratio_1'] = volume_eroded/volume_before
                    erode_df.loc[patient, 'volume_ratio_2'] = volume_eroded/volume_after
                    break
        else:
            erode_df.loc[patient, 'distance'] = 0
    erode_df.to_csv(f'/home/acy/data/lung/try/erode_230904_{args.threshold}_.csv')
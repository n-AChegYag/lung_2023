import os
import json
import SimpleITK as sitk
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/acy/data/lung/src')
from segmentation_metrics import compute_segmentation_scores
from tqdm import tqdm
from skimage import measure
from scipy import spatial

class SegmentationMetricsCalculator():

    def __init__(self, spacing=(1,1,5)):
        self.spacing = spacing
        self.BD = sitk.BinaryDilateImageFilter()
        self.BD.SetKernelType(sitk.sitkBall)
        self.BD.SetKernelRadius(1)
        self.BD.SetForegroundValue(1)
        self.BD.SetBackgroundValue(0)
        self.pred = None
        self.pred_111 = None
        self.pred_array = None
        self.pred_111_array = None
        self.target = None
        self.target_111 = None
        self.target_array = None
        self.target_111_array = None
        self.tumor = None
        self.tumor_111 = None
        self.tumor_array = None
        self.tumor_111_array = None
        self.tumor_prob = None
        self.tumor_prob_111 = None
        self.tumor_prob_array = None
        self.tumor_prob_111_array = None

    def compute_volume_ratio_and_center_distance(self):
        pred_props = measure.regionprops(self.pred_array)
        target_props = measure.regionprops(self.target_array)
        pred_volume = pred_props[0].area
        target_volume = target_props[0].area
        volume_ratio = pred_volume/target_volume
        pred_center_cood = np.array(pred_props[0].centroid)
        target_center_cood = np.array(target_props[0].centroid)
        center_distance = np.linalg.norm((pred_center_cood - target_center_cood) * np.array(self.spacing[::-1]), ord=2)
        return volume_ratio, center_distance
    
    def compute_dilate_size_when_recall_equal_1(self, threshold=0.90):
        recall, _ = self.compute_PR(ori_or_dilated='ori')
        if recall >= threshold:
            return 0
        else:
            for i in range(1,25):
                self.BD.SetKernelRadius(i)
                # self.pred_111_dilated = self.BD.Execute(self.pred_111)
                self.pred_111_dilated = self.BD.Execute(sitk.Cast(self.pred_111, sitk.sitkInt32))
                self.pred_111_dilated_array = sitk.GetArrayFromImage(self.pred_111_dilated)
                recall, _ = self.compute_PR(ori_or_dilated='dilated')
                if recall >= threshold:
                    break
            return i

    def calc_tumor_diameter(self):
        # by Li Ruikun
        tumor_region = measure.label(self.tumor_111_array, background=0)
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

    def compute_PR(self, ori_or_dilated='ori'):
        if ori_or_dilated == 'ori':
            pred = self.pred_111_array
        elif ori_or_dilated == 'dilated':
            pred = self.pred_111_dilated_array
        if np.sum(pred) == 0:
            recall = 0
            precision = 0
        else:
            fp = np.sum(np.logical_and(np.logical_not(self.target_111_array), pred))
            tp = np.sum(np.logical_and(pred, self.target_111_array))
            fn = np.sum(np.logical_and(np.logical_not(pred), self.target_111_array))
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
        return recall, 1 - precision
    
    @staticmethod
    def resample_data(data, out_spacing, image_or_label):
        # data is sitk image
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
    
    DATA_PATH = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/cl'
    PRED_PATH = '/home/acy/data/lung/log/inference_for_nnunet_a/output'
    TUMOR_PRED_PATH = '/home/acy/data/lung/log/inference_for_nnunet_b/output'
    PKL_PATH = '/home/acy/data/lung/src/PL/splits/splits_cl_231012_mc_c_16.pkl'
    with open(PKL_PATH) as f:
        train_valid_split = json.load(f)
        test_patients = train_valid_split['test']
    patients = [patient.split('/')[-1] for patient in test_patients]

    SPACING = (1,1,5)
    SMC = SegmentationMetricsCalculator(spacing=SPACING)

    metrics_df = pd.DataFrame(index=patients, columns=[
        'vr', 
        'cd', 
        'dice', 
        'hd95', 
        'recall', 
        'precision', 
        'd',
        'ds', 
        'reduce_rate_ori', 
        'reduce_rate_dilate',
        'CR', 
        'IER', 
    ])

    for patient in tqdm(patients):
        try:
            # load data
            SMC.pred = sitk.ReadImage(os.path.join(PRED_PATH, f'{patient}.nii.gz'))
            SMC.pred_111 = SMC.resample_data(SMC.pred, (1,1,1), 'label')
            SMC.pred_array = sitk.GetArrayFromImage(SMC.pred).astype(np.int64)
            SMC.pred_111_array = sitk.GetArrayFromImage(SMC.pred_111).astype(np.int64)

            SMC.target = sitk.ReadImage(os.path.join(DATA_PATH, patient, 'label_a.nii.gz'))
            SMC.target_111 = SMC.resample_data(SMC.target, (1,1,1), 'label')
            SMC.target_array = sitk.GetArrayFromImage(SMC.target)#.astype(np.int64)
            SMC.target_111_array = sitk.GetArrayFromImage(SMC.target_111)#.astype(np.int64)

            SMC.tumor = sitk.ReadImage(os.path.join(DATA_PATH, patient, 'label_b.nii.gz'))
            SMC.tumor_111 = SMC.resample_data(SMC.tumor, (1,1,1), 'label')
            SMC.tumor_array = sitk.GetArrayFromImage(SMC.tumor).astype(np.int64)
            SMC.tumor_111_array = sitk.GetArrayFromImage(SMC.tumor_111).astype(np.int64)

            SMC.tumor_prob = sitk.ReadImage(os.path.join(TUMOR_PRED_PATH, f'{patient}.nii.gz'))
            SMC.tumor_prob_111 = SMC.resample_data(SMC.tumor_prob, (1,1,1), 'image')
            SMC.tumor_prob_array = sitk.GetArrayFromImage(SMC.tumor_prob)
            SMC.tumor_prob_111_array = sitk.GetArrayFromImage(SMC.tumor_prob_111)

            tumor_pred_array = np.zeros_like(SMC.tumor_prob_array)
            tumor_pred_array[SMC.tumor_prob_array >= 0.5] = 1
            tumor_pred_111_array = np.zeros_like(SMC.tumor_prob_111_array)
            tumor_pred_111_array[SMC.tumor_prob_111_array >= 0.5] = 1

            # SMC.pred_array = np.logical_and(tumor_pred_array, SMC.pred_array)*1
            # SMC.pred_111_array = np.logical_and(tumor_pred_111_array, SMC.pred_111_array)*1

            # compute metrics
            volume_ratio, center_distance = SMC.compute_volume_ratio_and_center_distance()
            seg_metrics = compute_segmentation_scores(SMC.target_array, SMC.pred_array, SPACING)
            # seg_metrics = compute_segmentation_scores(SMC.target_array, SMC.tumor_array, SPACING)
            # seg_metrics = compute_segmentation_scores(SMC.pred_array, SMC.tumor_array, SPACING)
            d = SMC.calc_tumor_diameter()
            dilate_size = SMC.compute_dilate_size_when_recall_equal_1()
            if dilate_size != 0:
                SMC.BD.SetKernelRadius(dilate_size)
                # SMC.pred_111 = SMC.BD.Execute(SMC.pred_111)
                SMC.pred_111 = SMC.BD.Execute(sitk.Cast(SMC.pred_111, sitk.sitkInt32))
                SMC.pred_111_array = sitk.GetArrayFromImage(SMC.pred_111).astype(np.int64)
            corrected_dilated_pred_111_array = np.logical_and(tumor_pred_111_array, SMC.pred_111_array)*1
            reduce_ori =  np.sum(SMC.pred_array) / np.sum(SMC.tumor_array)
            reduce_dilate = np.sum(corrected_dilated_pred_111_array) / np.sum(SMC.tumor_111_array)
            CR, IER = SMC.compute_PR()

            # record
            metrics_df.loc[patient, 'vr'] = volume_ratio
            metrics_df.loc[patient, 'cd'] = center_distance
            metrics_df.loc[patient, 'dice'] = seg_metrics['dice_score']
            metrics_df.loc[patient, 'hd95'] = seg_metrics['hausdorff_distance_95']
            metrics_df.loc[patient, 'recall'] = seg_metrics['recall']
            metrics_df.loc[patient, 'precision'] = seg_metrics['precision']
            metrics_df.loc[patient, 'd'] = d
            metrics_df.loc[patient, 'ds'] = dilate_size
            metrics_df.loc[patient, 'reduce_rate_ori'] = reduce_ori
            metrics_df.loc[patient, 'reduce_rate_dilate'] = reduce_dilate
            metrics_df.loc[patient, 'CR'] = CR
            metrics_df.loc[patient, 'IER'] = IER
        except IndexError:
            continue

    metrics_df.to_csv(f'/home/acy/data/lung/try/231105_corrected_0.001_nnunet.csv')
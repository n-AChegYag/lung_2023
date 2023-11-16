import os
import json
import torch
import argparse
import transforms
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
from lung_dataset import LungDataset, LungSegDataset

import sys
sys.path.append('/home/acy/data/lung/src/')
from model.my_model import MyEncoder, MyDecoder


class Predictor:

    def __init__(self,
                 model,
                 path_to_model_weights,  # list of paths or path
                 dataloader,
                 path_to_save_dir='.'
                 ):

        self.model_E = model['E']
        self.model_D = model['D']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_to_model_weights = [os.path.join(path_to_model_weights, weights_name) for weights_name in os.listdir(path_to_model_weights)] if os.path.isdir(path_to_model_weights) else path_to_model_weights
        self.dataloader = dataloader
        self.path_to_save_dir = path_to_save_dir

    def predict(self):
        """Run inference for an single model"""

        if self.device.type == 'cpu':
            print(f'Run inference for a model on CPU')
        else:
            print(f'Run inference for a model'
                  f' on {torch.cuda.get_device_name(torch.cuda.current_device())}')

        # Check if the directory exists:
        if not os.path.exists(self.path_to_save_dir):
            os.makedirs(self.path_to_save_dir, exist_ok=True)

        # Send model to device:
        self.model_E = self.model_E.to(self.device)
        self.model_E.eval()
        self.model_D = self.model_D.to(self.device)
        self.model_D.eval()

        # Load model weights:
        model_weights = torch.load(self.path_to_model_weights, map_location=torch.device(self.device))
        self.model_E = self._load_model_weights(model_weights, self.model_E, 'E')
        self.model_D = self._load_model_weights(model_weights, self.model_D, 'D')

        # Inference:
        with torch.no_grad():
            for sample, info in tqdm(self.dataloader):
                input = sample['input']
                input = input.to(self.device)
                x, ds0, ds1, ds2 = self.model_E(input)
                logit = self.model_D(x, ds0, ds1, ds2)
                prob = torch.sigmoid(logit)
                if self.device.type == "cuda":
                    sample['output'] = prob.cpu().numpy()
                elif self.device.type == "cpu":
                    sample['output'] = prob.numpy()

                # Save prediction:
                self._save_preds(sample, info, self.path_to_save_dir)

        print(f'Predictions have been saved in {self.path_to_save_dir}')
        
    def ensemble_predict(self):

        """Run inference for an ensemble of models"""

        if self.device.type == 'cpu':
            print(f'Run inference for a model on CPU')
        else:
            print(f'Run inference for a model'
                  f' on {torch.cuda.get_device_name(torch.cuda.current_device())}')

        # Check if the directory exists:
        if not os.path.exists(self.path_to_save_dir):
            os.makedirs(self.path_to_save_dir, exist_ok=True)

        # Send model to device:
        self.model_E = self.model_E.to(self.device)
        self.model_E.eval()
        self.model_D = self.model_D.to(self.device)
        self.model_D.eval()

        # Inference:
        with torch.no_grad():
            for sample, info in tqdm(self.dataloader):
                input = sample['input']
                input = input.to(self.device)
                ensemble_prob = 0
                for weight_path in self.path_to_model_weights:
                    # Load model weights:
                    model_weights = torch.load(weight_path, map_location=torch.device(self.device))
                    self.model_E = self._load_model_weights(model_weights, self.model_E, 'E')
                    self.model_D = self._load_model_weights(model_weights, self.model_D, 'D')
                    x, ds0, ds1, ds2 = self.model_E(input)
                    logit = self.model_D(x, ds0, ds1, ds2)
                    prob = torch.sigmoid(logit)
                    ensemble_prob += prob
                ensemble_prob /= len(self.path_to_model_weights)
                
                if self.device.type == "cuda":
                    sample['output'] = ensemble_prob.cpu()
                elif self.device.type == "cpu":
                    sample['output'] = ensemble_prob

                # Save prediction:
                self._save_preds(sample, info, self.path_to_save_dir)

        print(f'Predictions have been saved in {self.path_to_save_dir}')

    @staticmethod
    def _save_preds(sample, info, path_to_dir, threshold=0.5):
        prob = sample['output']
        sample_id = info['patient_id_'][0]
        bbox = info['bbox']
        ori_size = info['ori_size']
        pred = np.zeros((ori_size[0].item(), ori_size[1].item(), ori_size[2].item()))
        prob_ = np.zeros((ori_size[0].item(), ori_size[1].item(), ori_size[2].item()))
        prob_[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = prob
        pred[prob_>=threshold] = 1
        pred[prob_<threshold]  = 0
        pred = sitk.GetImageFromArray(pred)
        prob_ = sitk.GetImageFromArray(prob_)
        pred.SetSpacing((1,1,5))
        prob_.SetSpacing((1,1,5))
        if not os.path.exists(os.path.join(path_to_dir, sample_id)):
            os.makedirs(os.path.join(path_to_dir, sample_id))
        sitk.WriteImage(pred, os.path.join(path_to_dir, sample_id, 'pred.nii.gz'))
        sitk.WriteImage(prob_, os.path.join(path_to_dir, sample_id, 'prob.nii.gz'))

    @staticmethod
    def _load_model_weights(pl_ckpt, model, model_name):
        from collections import OrderedDict
        new_model_state_dict = OrderedDict()
        for k,v in pl_ckpt['state_dict'].items():
            if model_name[0] == k[6:7]:
                new_model_state_dict[k[8:]] = v
        model.load_state_dict(new_model_state_dict, strict=True)
        return model

def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # parameters
    num_workers = 4
    n_filters = 32
    path_to_pkl = '/home/acy/data/lung/src/PL/splits/splits_cl_231012_mc_c_16.pkl'

    with open(path_to_pkl) as f:
        train_valid_split = json.load(f)
        test_paths = train_valid_split['test']
        
    input_transforms = transforms.Compose([
    transforms.NormalizeIntensity(),
    transforms.ToTensor(mode='test')
    ])
    
    # dataset and dataloader:
    if args.mode == 'pre':
        data_set = LungSegDataset(test_paths, 'test', transforms=input_transforms)
    elif args.mode == 'trans':
        data_set = LungDataset(test_paths, 'test', transforms=input_transforms)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=num_workers)
    
    # model
    if args.mode == 'pre':
        model = {
            'E':  MyEncoder(1,n_filters,2),
            'D': MyDecoder(1,n_filters,2)
        }
    elif args.mode == 'trans':
        model = {
            'E':  MyEncoder(2,n_filters,2),
            'D': MyDecoder(2,n_filters,2)
        }  
        
    predictor = Predictor(model, args.weights_path, data_loader, args.save_path)
    if os.path.isdir(args.weights_path):
        predictor.ensemble_predict()
    else:
        predictor.predict()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Model Inference Script')
    parser.add_argument('-g', '--gpus', default='2', type=str, help='Index of GPU used')
    parser.add_argument('-m', '--mode', default='trans', type=str, help='trans or pre')
    parser.add_argument('-w', '--weights_path', required=False, default='/home/acy/data/lung/log/231105_trans_b2_seg_32_0.001/lightning_logs/version_0/checkpoints', type=str, help='path to features')
    parser.add_argument('-s', '--save_path', required=False, default='/home/acy/data/lung/log/231105_trans_b2_seg_32_0.001/lightning_logs/version_0/pred', type=str, help='path to pkl file of data splits')
    args = parser.parse_args()

    import time
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('time: {}s'.format(end_time-start_time))
import os
import json
import torch
import pickle
import argparse
import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
from lung_dataset import LungRadiomicsDataset, LungRadiomicsClinicalDataset, LungClinicalDataset

import sys
sys.path.append('/home/acy/data/lung/src/')
from model.my_model import MyEncoder, MyClassifer


class Predictor:

    def __init__(self,
                 model,
                 path_to_model_weights,  # list of paths or path
                 dataloader,
                 path_to_save_dir='.'
                 ):

        self.model_E = model['E']
        self.model_classifer = model['classifer']
        self.m = torch.nn.Softmax(dim=1)
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
        self.model_classifer = self.model_classifer.to(self.device)
        self.model_classifer.eval()

        # Load model weights:
        model_weights = torch.load(self.path_to_model_weights, map_location=torch.device(self.device))
        self.model_E = self._load_model_weights(model_weights, self.model_E, 'E')
        self.model_classifer = self._load_model_weights(model_weights, self.model_classifer, 'classifer')

        # Inference:
        with torch.no_grad():
            for sample, info in tqdm(self.dataloader):
                input = sample['input']
                feats = sample['feature']
                input = input.to(self.device)
                feats = feats.to(self.device)
                x, _, _, _ = self.model_E(input)
                class_logit = self.model_classifer(x, feats)
                class_prob = self.m(class_logit)
                
                if self.device.type == "cuda":
                    sample['class_pred'] = class_prob.cpu()
                elif self.device.type == "cpu":
                    sample['class_pred'] = class_prob

                # Save prediction:
                self._save_preds(sample, info, self.path_to_save_dir)

        print(f'Predictions have been saved in {self.path_to_save_dir}')

    @staticmethod
    def _save_preds(sample, info, path_to_dir):
        class_prob = sample['class_pred']
        class_pred = class_prob.argmax(dim=1).numpy()[0]
        sample_id = info['patient_id_'][0]
        if not os.path.exists(os.path.join(path_to_dir, sample_id + f'_{class_pred}')):
            os.makedirs(os.path.join(path_to_dir, sample_id + f'_{class_pred}'))
        with open(os.path.join(path_to_dir, sample_id + f'_{class_pred}', 'prob.pth'), 'wb') as f:
            pickle.dump(class_prob.numpy(), f)

    @staticmethod
    def _load_model_weights(pl_ckpt, model, model_name):
        from collections import OrderedDict
        new_model_state_dict = OrderedDict()
        for k,v in pl_ckpt['state_dict'].items():
            if model_name[0] == k[6:7]:
                if model_name == 'E':
                    new_model_state_dict[k[8:]] = v
                elif model_name == 'classifer':
                    new_model_state_dict[k[16:]] = v
        model.load_state_dict(new_model_state_dict, strict=True)
        return model

def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # parameters
    num_workers = 4
    n_filters = 32
    path_to_pkl = '/home/acy/data/lung_2023/data/splits/splits.pkl'
    path_to_info = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/info_1107.xlsx'
    weights_path = f'/home/acy/data/lung_2023/logs/pre_cls_{args.feats}/weights/weights_fake.ckpt'
    save_path = f'/home/acy/data/lung_2023/logs/pre_cls_{args.feats}/pred_fake/pred_{args.part}'

    with open(path_to_pkl) as f:
        train_valid_split = json.load(f)
        valid_paths = train_valid_split[args.part]
        
    input_transforms = transforms.Compose([
    transforms.NormalizeIntensity(),
    transforms.ToTensor(mode='test')
    ])
    
    # dataset and dataloader:
    if args.feats == 'r':
        data_set = LungRadiomicsDataset(valid_paths, path_to_info, 'test', transforms=input_transforms, gt='pred')
    elif args.feats == 'cr':
        data_set = LungRadiomicsClinicalDataset(valid_paths, path_to_info, 'test', transforms=input_transforms, gt='pred')
    elif args.feats == 'c':
        data_set = LungClinicalDataset(valid_paths, path_to_info, 'test', transforms=input_transforms, gt='pred')
    elif args.feats == 'cnn':
        data_set = LungRadiomicsClinicalDataset(valid_paths, path_to_info, 'test', transforms=input_transforms, gt='pred')
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=num_workers)
    
    # model
    if args.feats == 'r':
        model = {
            'E':  MyEncoder(2,n_filters,2),
            'classifer': MyClassifer(n_filters, 10, 2)
        }
    elif args.feats == 'c':
        model = {
            'E':  MyEncoder(2,n_filters,2),
            'classifer': MyClassifer(n_filters, 14, 2)
        }
    elif args.feats == 'cr':
        model = {
            'E':  MyEncoder(2,n_filters,2),
            'classifer': MyClassifer(n_filters, 24, 2)
        }     
    elif args.feats == 'cnn':
        model = {
            'E':  MyEncoder(2,n_filters,2),
            'classifer': MyClassifer(n_filters, 0, 2)
        }  
        
    predictor = Predictor(model, weights_path, data_loader, save_path)
    predictor.predict()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Model Inference Script')
    parser.add_argument('-g', '--gpus', default='0', type=str, help='Index of GPU used')
    parser.add_argument('-p', '--part', default='test', type=str, help='train set, validation set or test set')
    parser.add_argument("-fe", "--feats",  type=str, required=False, default='cnn', choices=['r', 'c', 'cr', 'cnn'], help="which splits")
    args = parser.parse_args()

    import time
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('time: {}s'.format(end_time-start_time))
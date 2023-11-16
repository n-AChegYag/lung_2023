import os
import json
import random
import argparse

def archive_centers(ids):
    center_dict = {
        'center_1':  [],
        'center_2':  [],
        'center_3':  [],
        'center_4':  [],
        'center_5':  [],
        'center_6':  []
    }
    for id in ids:
        if id[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            center_dict['center_1'].append(id)
        elif id[0:2] == 'cq' or id[0:2] == 'CQ':
            center_dict['center_2'].append(id)
        elif id[0:2] == 'js':
            center_dict['center_3'].append(id)
        elif id[0:3] == 'lyg' or id[0:3] == 'Lyg':
            center_dict['center_4'].append(id)
        elif id[0:2] == 'sx' or id[0:2] == 'sy':
            center_dict['center_5'].append(id)
        elif id[0:4] == 'lung' or id[0:4] == 'Lung': 
            center_dict['center_1'].append(id)
        else:
            center_dict['center_6'].append(id)
    return center_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, required=False, default=666)
    args = parser.parse_args()
    
    train_test_split = [0.84, 0.16]
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical'
    save_path = '/home/acy/data/lung/src/PL/splits'
    tag = '231012_mc_c'

    cl_ids = os.listdir(os.path.join(data_path, 'cl'))
    xt_ids = os.listdir(os.path.join(data_path, 'xt'))
    
    cl_center_dict = archive_centers(cl_ids)
    xt_center_dict = archive_centers(xt_ids)
    
    cl_train_ids, cl_test_ids = [], []
    cl_train_ids_2, cl_test_ids_2 = [], []
    xt_train_ids, xt_test_ids = [], []
    xt_train_ids_2, xt_test_ids_2 = [], []
    
    for center, ids in cl_center_dict.items():
        if center == 'center_3':
            continue
        cl_train_ids.extend(ids)
        random.seed(args.seed)
        random.shuffle(ids)
        cl_train_ids_2.extend(ids[0:int(len(ids)*train_test_split[0])])
        cl_test_ids_2.extend(ids[int(len(ids)*train_test_split[0]):])
    cl_test_ids = cl_center_dict['center_3']
    cl_test_ids_2.extend(cl_center_dict['center_3'])

    for center, ids in xt_center_dict.items():
        if center == 'center_3':
            continue
        xt_train_ids.extend(ids)
        random.seed(args.seed)
        random.shuffle(ids)
        xt_train_ids_2.extend(ids[0:int(len(ids)*train_test_split[0])])
        xt_test_ids_2.extend(ids[int(len(ids)*train_test_split[0]):])
    xt_test_ids = xt_center_dict['center_3']
    xt_test_ids_2.extend(xt_center_dict['center_3'])
        
    cl_tag = os.path.join(data_path, 'cl')
    xt_tag = os.path.join(data_path, 'xt')
    
    cl_train_list = [os.path.join(cl_tag, id) for id in cl_train_ids]
    cl_train_list_2 = [os.path.join(cl_tag, id) for id in cl_train_ids_2]
    cl_test_list = [os.path.join(cl_tag, id) for id in cl_test_ids]
    cl_test_list_2 = [os.path.join(cl_tag, id) for id in cl_test_ids_2]
    
    xt_train_list = [os.path.join(xt_tag, id) for id in xt_train_ids]
    xt_train_list_2 = [os.path.join(xt_tag, id) for id in xt_train_ids_2]
    xt_test_list = [os.path.join(xt_tag, id) for id in xt_test_ids]
    xt_test_list_2 = [os.path.join(xt_tag, id) for id in xt_test_ids_2]
    
    train_list = cl_train_list + xt_train_list
    train_list_2 = cl_train_list_2 + xt_train_list_2
    test_list = cl_test_list + xt_test_list
    test_list_2 = cl_test_list_2 + xt_test_list_2
    random.seed(args.seed)
    random.shuffle(train_list_2)
    valid_list_2 = train_list_2[-11:]
    train_list_2 = train_list_2[:-11]
    
    
    data_split_dict = {
        'train': train_list,
        'test': test_list
    }
    cl_data_split_dict = {
        'train': cl_train_list,
        'test': cl_test_list
    }
    data_split_dict_2 = {
        'train': train_list_2,
        'valid': valid_list_2,
        'test': test_list_2
    }
    cl_data_split_dict_2 = {
        'train': cl_train_list_2,
        'valid': valid_list_2,
        'test': cl_test_list_2
    }
    
    with open(os.path.join(save_path, f'splits_{tag}.pkl'), 'w') as f:
        json.dump(data_split_dict, f)
    with open(os.path.join(save_path, f'splits_cl_{tag}.pkl'), 'w') as f:
        json.dump(cl_data_split_dict, f)
    with open(os.path.join(save_path, f'splits_{tag}_{args.seed}.pkl'), 'w') as f:
        json.dump(data_split_dict_2, f)
    with open(os.path.join(save_path, f'splits_cl_{tag}_{args.seed}.pkl'), 'w') as f:
        json.dump(cl_data_split_dict_2, f)
        
    
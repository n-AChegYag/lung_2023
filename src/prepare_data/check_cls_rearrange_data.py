import os

if __name__ == '__main__':
    
    data_path = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample'
    tags = ['cl', 'xt']
    
    for tag in tags:
        print(tag)
        for patient in os.listdir(os.path.join(data_path, tag)):
            if tag == 'cl':
                if len(os.listdir(os.path.join(data_path, tag, patient))) != 3:
                    print(f'len of {patient} is {len(os.listdir(os.path.join(data_path, tag, patient)))}')
            elif tag == 'xt':
                if len(os.listdir(os.path.join(data_path, tag, patient))) != 2:
                    print(f'len of {patient} is {len(os.listdir(os.path.join(data_path, tag, patient)))}')
                
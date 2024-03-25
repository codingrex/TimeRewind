import numpy as np
import os 

ROOT_DIR= '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/01_train_controlNet_All'

def convert(root_dir):
    save_dic= {}
    save_dic['t0']= []
    save_dic['t1']= []
    save_dic['event']= []


    for key in save_dic.keys():
        folder= os.path.join(ROOT_DIR, key)
        folder_content= sorted(os.listdir(folder))

        for i in range(len(folder_content)):
            file_name= folder_content[i]
            file_name= os.path.join(folder, file_name)
            save_dic[key].append(file_name)
        
    
    # save_dic['text']= ['High Quality Image'] * len(save_dic['t0'])
    save_dic['text']= [''] * len(save_dic['t0'])
    

    save_dic_path= os.path.join(root_dir, 'dataset.npy')

    np.save(save_dic_path, save_dic)

    test_save= np.load(save_dic_path, allow_pickle= True)

    print(test_save)



def main():
    convert(ROOT_DIR)



if __name__ == '__main__':
    main()

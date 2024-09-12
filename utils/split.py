import random
import shutil
import glob
import os
from sklearn.model_selection import train_test_split


train_ratio = 0.60
val_ratio = 0.2
test_ratio = 0.2

folders = ['train', 'valid', 'test']



def DeleteCreateFolders()-> None:
    #Delete and Create folders
    for folder in folders:
        save_folder = os.path.join(dir, folder)
        try:
            shutil.rmtree(save_folder)
        except OSError:
            print('Previous dataset not found')

        os.makedirs(save_folder)


def isPathValid(dir:str)-> bool:
    paths = []
    for folder in folders[:2]:
        im_dir = dir + '/' + folder + '_images'
        mask_dir = dir + '/' + folder + '_masks'
    
        assert os.path.isdir(im_dir), "Dataset cannot found"
        assert os.path.isdir(mask_dir), "Dataset cannot found"
        paths.append(im_dir)
        paths.append(mask_dir)
    return paths

def collect2folder(paths):
    dest_dir = "dataset/train"
    train_im = []
    train_masks = []
    j = 0
    for path in paths:
        if 'masks' in path:
            file_list = glob.glob(path + '/*')
            for file in file_list:
                base, extension = os.path.splitext(file)
                base = os.path.basename(base)
                dest = dest_dir + '/' + base + '_' + str(j-1) + '_mask' + extension
                shutil.copy(file, dest)
                train_masks.append(dest)
        else:
            file_list = glob.glob(path + '/*')
            for file in file_list:
                base, extension = os.path.splitext(file)
                base = os.path.basename(base)
                dest = dest_dir + '/' + base + '_' + str(j) + extension
                shutil.copy(file, dest)
                train_im.append(dest)
        j += 1

    return train_im, train_masks

def split_directories(paths:str, folders:list[str])-> None:

    '''
        builds dataset directories distributes data randomly with needed ratio
        
        Args:
        -----
            dir : str
                root directory of dataset
            labels : list
                it consists of sub-directories(which means labels) of dataset

        Returns:
        --------
            None
        '''

    DeleteCreateFolders()

    img_list, img_masks = collect2folder(paths)
    img_list.sort()
    img_masks.sort()


    trainX, testX, trainY, testY  = train_test_split(img_list, img_masks, test_size=val_ratio+test_ratio, shuffle=False)

    validX, testX, validY, testY = train_test_split(testX, testY, test_size=test_ratio/(val_ratio+test_ratio), shuffle=False)
    
    paths_all = [(trainX, trainY), (validX, validY), (testX, testY)]

    #create folders and copy and send all data where needs
    for (folder, paths) in zip(folders[1:3], paths_all[1:3]):
        save_folder = os.path.join(dir, folder)
        for i in range(len(paths[0])):
                shutil.move(paths[0][i], save_folder)
                shutil.move(paths[1][i], save_folder)


if __name__ == "__main__":
    #Test the validity of path
    dir = "dataset"
    paths = isPathValid(dir)
    split_directories(paths, folders)



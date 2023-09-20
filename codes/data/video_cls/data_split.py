import os
from shutil import copyfile
from tqdm import tqdm

new_folder = os.path.join('D:','vivit','mydata')

# 파일명 rename하기 
for folder in ['accident','driving']:
    for i, file in enumerate(os.listdir(os.path.join(new_folder,folder))):
        curr_name = os.path.join(new_folder,folder,file)
        new_name = os.path.join(new_folder,folder,folder+f'_00{i}.mp4')
        os.rename(curr_name,new_name)

# train, val, test 폴더 만들기
for folder in ['train','val','test']:
    for sub_folder in ['accident','driving']:
        folder_path = os.path.join(new_folder,folder,sub_folder)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        print('pass')

# train, val, test 채우기
train_cnt = 0.7 * 60
val_cnt = 0.1 * 60
test_cnt = 0.2 * 60

for folder in tqdm(['accident', 'driving']):
    folder_path = os.path.join(new_folder, folder)
    cnt = 0
    for image in os.listdir(folder_path):
        curr_path = os.path.join(folder_path, image)
        if cnt < train_cnt:
            move_path = os.path.join(new_folder, 'train', folder, image)
        elif cnt < train_cnt + val_cnt:
            move_path = os.path.join(new_folder, 'val', folder, image)
        else:
            move_path = os.path.join(new_folder, 'test', folder, image)

        copyfile(curr_path, move_path)
        cnt += 1
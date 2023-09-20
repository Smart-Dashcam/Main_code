import os
import re
import pickle
import xmltodict
import pandas as pd
from shutil import copyfile
from distutils.dir_util import copy_tree

folder_path = 'D:/도로주행'
new_folder = "D:/yolo_dataset2"
if not os.path.isdir(new_folder):
    os.mkdir(new_folder)

#############################################################################################
# TASK 1. 각 폴더들 교집합 없는지 확인. image, label 맞지 않는 파일 제거
#############################################################################################
image_folder, label_folder = [],[]
image_set, label_set = set(), set()
image_cnt, label_cnt = 0, 0

if not os.path.isfile(new_folder + '/task2_1_complete.txt'):
    for folder in os.listdir(folder_path):
        ff_path = os.path.join(folder_path,folder)
        ftype = folder.split('_')[-1]
        files = [x.split('.')[0] for x in os.listdir(ff_path)]
        if ftype == '이미지':
            image_set.update(files)
            image_cnt += len(files)
            image_folder.append([folder, len(files)])
        elif ftype == '라벨':
            files = [x.replace('_v001_1', '') for x in files]
            label_set.update(files)
            label_cnt += len(files)
            label_folder.append([folder, len(files)])

    image_folder.append(image_cnt)
    label_folder.append(label_cnt)

    print(f'이미지 파일 정보 {image_folder}')
    print(f'라벨 파일 정보 {label_folder}')

    print(f'이미지 통합 후 파일 수 {len(image_set)}')
    print(f'라벨 통합 후 파일 수 {len(label_set)}')

    remove_image = image_set - label_set
    print(f'제거 대상 파일(이미지): {remove_image}')     # 1개라서 수작업으로 제거함.

    with open(new_folder + '/task2_1_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 2. image, label 폴더로 통합하기
#############################################################################################
if not os.path.isfile(new_folder + '/task2_2_complete.txt'):
    # 통합하기 위한 폴더 생성
    flist = ['images','labels']
    for f in flist:
        if not os.path.isdir(os.path.join(new_folder,f)):
            os.mkdir(os.path.join(new_folder,f))
            print(f'save {os.path.join(folder_path,f)}')

    # 이미지 이동시키기
    for folder in os.listdir(folder_path):
        if folder in ['images', 'labels']:
            continue
        move_path = ''
        ftype = folder.split('_')[-1]
        curr_path = os.path.join(folder_path,folder)
        if ftype == '이미지':
            move_path = os.path.join(new_folder,'images')
        elif ftype == '라벨':
            move_path = os.path.join(new_folder,'labels')
        copy_tree(curr_path, move_path)
        print(f'> {folder} 이동 완료', end=' ')
    print()
    with open(new_folder + '/task2_2_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 3. class count 알아보기 (필요없는 class 제거하기 위함)
#############################################################################################
classes = ['none_of_the_above','Vehicle_Car','Vehicle_Bus','Vehicle_Motorcycle','Vehicle_Unknown',
           'Pedestrian_Pedestrian','Pedestrian_Bicycle',
           'Lane_White_Dash','Lane_White_Solid','Lane_Yellow_Dash','Lane_Yello_Solid','Lane_Blue_Dash','Lane_Blue_Solid',
           'TrafficLight_Red','TrafficLight_Yellow','TrafficLight_Green','TrafficLight_Arrow','TrafficLight_RedArrow','TrafficLight_YellowArrow','TrafficLight_GreenArrow',
           'TrafficSign_Speed','TrafficSign_Else',
           'RoadMark_StopLine','RoadMark_Crosswalk','RoadMark_Number','RoadMark_Character',
           'RoadMarkArrow_Straight','RoadMarkArrow_Left','RoadMarkArrow_Right','RoadMarkArrow_StraightLeft','RoadMarkArrow_StraightRight','RoadMarkArrow_Uturn','RoadMarkArrow_Else',
           'FreeSpace']
class_cnt = {x:0 for x in classes}

def count_class(xml_path):
    global class_cnt

    with open(xml_path, 'r', encoding='utf-8') as f:
        xmlfile = f.read()

    xml2json = xmltodict.parse(xmlfile)['annotation']
    boxes = xml2json['object']
    if type(boxes) == dict:
        boxes = [boxes]

    for box in boxes:
        box_class = box['name']
        class_cnt[box_class] += 1

if not os.path.isfile(new_folder + '/task2_3_complete.txt'):
    print('Start Count!', end=' ')
    label_path = os.path.join(new_folder, 'labels')
    i = 0
    for file in os.listdir(label_path):
        if file.split('.')[-1] == 'txt':
            continue
        xml_path = os.path.join(label_path, file)
        count_class(xml_path)

        i += 1
        if i % 2000 == 0:
            print(f'> {i}', end=' ')
    print('>>> End!')

with open(new_folder + '/class_cnt.pkl', 'wb') as f:
    pickle.dump(class_cnt, f)

with open(new_folder + '/task2_3_complete.txt','w') as f:
    f.write('complete')

print(class_cnt)
# print(len(sorted([(k,v) for k,v in class_cnt.items() if v != 0],key=lambda x:x[1])))
# print(sorted([(k,v) for k,v in class_cnt.items() if v != 0],key=lambda x:x[1]))

# Select Class(11)
# Vehicle_Car, Vehicle_Bus, Vehicle_Motorcycle, Vehicle_Unknown,
# Pedestrian_Pedestrian, Pedestrian_Bicycle,
# TrafficLight_Red, TrafficLight_Yellow, TrafficLight_Green,
# RoadMark_StopLine, RoadMark_Crosswalk

#############################################################################################
# TASK 4. xml 파일을 json 파일로 바꾼 후 txt 만들기
#############################################################################################
new_classes = ['Vehicle_Car','Vehicle_Bus','Vehicle_Motorcycle','Vehicle_Unknown',
               'Pedestrian_Pedestrian','Pedestrian_Bicycle',
               'TrafficLight_Red','TrafficLight_Yellow','TrafficLight_Green',
               'RoadMark_StopLine','RoadMark_Crosswalk']
class_info = {x:i for i, x in enumerate(new_classes)}

def box_normalize(bbox, imgsize):
    w, h = imgsize
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    xc = x1 + width / 2
    yc = y1 + height / 2

    return [xc/w, yc/h, width/w, height/h]

def xml2txt(xml_path):
    global class_cnt

    with open(xml_path,'r',encoding='utf-8') as f:
        xmlfile = f.read()

    txt_path = xml_path.replace('_v001_1.xml','.txt')
    xml2json = xmltodict.parse(xmlfile)['annotation']
    size = list(map(int,xml2json['size'].values()))[:-1]
    boxes = xml2json['object']
    if type(boxes) == dict:
        boxes = [boxes]

    box_list = []
    for box in boxes:
        box_class = box['name']
        if box_class in class_info:
            box_num = class_info[box_class]
            box_point = list(map(int,box['bndbox'].values()))
            for point in box_point:
                if point < 0:
                    return 0
            box_point_norm = box_normalize(box_point, size)
            for point in box_point_norm:
                if point < 0:
                    return 0
            box_list.append([box_num] + box_point_norm)

    data = pd.DataFrame(box_list)
    data.to_csv(txt_path, sep=' ', header=False, index=False)

    return 1

if not os.path.isfile(new_folder + '/task2_4_complete.txt'):
    print('Start XML to TXT!', end=' ')
    use_files = {}
    label_path = os.path.join(new_folder, 'labels')
    i = 0
    for file in os.listdir(label_path):
        if file.split('.')[-1] == 'txt':
            continue
        xml_path = os.path.join(label_path,file)
        isuse = xml2txt(xml_path)
        use_path = file.replace('_v001_1.xml','')
        use_files[use_path] = isuse

        i += 1
        if i % 2000 == 0:
            print(f'> {i}', end=' ')
    print('>>> End!')

    with open(new_folder + '/use_files.pkl', 'wb') as f:
        pickle.dump(use_files, f)

    with open(new_folder + '/task2_4_complete.txt','w') as f:
        f.write('complete')

with open(new_folder + '/use_files.pkl', 'rb') as f:
    use_files = pickle.load(f)

print(len(use_files))
# print([k for k, v in use_files.items() if v == 0])
print(f'사용하지 않을 파일 {len([(k, v) for k, v in use_files.items() if v == 0])}개')

###############################################################################################
# TASK 5. 필요없는 파일 지우기
###############################################################################################
if not os.path.isfile(new_folder + '/task2_5_complete.txt'):
    for search in ['images', 'labels']:
        search_path = os.path.join(new_folder, search)
        for file in os.listdir(search_path):
            if file[-3:] == 'xml':
                continue
            if search == 'images':
                filename = file.replace('.jpg','')
            else:
                filename = file.replace('.txt','')
            # print(filename)
            if use_files[filename] == 0:
                os.remove(os.path.join(search_path,file))

    with open(new_folder + '/task2_5_complete.txt', 'w') as f:
        f.write('complete')
#############################################################################################
# TASK 6. train, valid, test 나누기
#############################################################################################
train_size = 0.7
valid_size = 0.1
test_size = 0.2

for name in ['train','valid','test']:
    for folder in ['images','labels']:
        find_path = os.path.join(new_folder, name, folder)
        if not os.path.isdir(find_path):
            os.makedirs(find_path)

if not os.path.isfile(new_folder + '/task2_6_complete.txt'):
    # train, valid, test 개수
    folders = os.listdir(new_folder + '/images')
    folder_n = len(folders)
    train_cnt = int(folder_n * train_size)
    valid_cnt = int(folder_n * valid_size)
    print(f'total_N={folder_n} train_N={train_cnt} valid_N={valid_cnt}')

    # 나눌 파일명 미리 설정하기
    folder_dict = {'train': [], 'valid': [], 'test': []}
    image_path = os.path.join(new_folder, 'images')
    for i, image in enumerate(os.listdir(image_path)):
        filename = image.split('.')[0]
        if i < train_cnt:
            folder_dict['train'].append(filename)
        elif i < train_cnt + valid_cnt:
            folder_dict['valid'].append(filename)
        else:
            folder_dict['test'].append(filename)

    # 파일 이동하기
    for search in ['images','labels']:
        print(f'{search}', end=' ')
        file_path = os.path.join(new_folder,search)
        cnt = 0
        for file in os.listdir(file_path):
            if file[-3:] == 'xml':
                continue
            curr_path = os.path.join(file_path, file)
            filename = file.split('.')[0]
            if search == 'label':
                filename = filename.replace('_v001_1','')
            if filename in folder_dict['train']:
                move_path = os.path.join(new_folder,'train',search,file)
            elif filename in folder_dict['valid']:
                move_path = os.path.join(new_folder,'valid',search,file)
            else:
                move_path = os.path.join(new_folder,'test',search,file)
            # print(curr_path)
            copyfile(curr_path, move_path)

            cnt += 1
            if cnt % 2000 == 0:
                print(f'> {cnt}', end=' ')
        print()


    with open(new_folder + '/task2_6_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 7. data.yaml 만들기
#############################################################################################
import yaml

data = {
    'train':'D:/yolo_dataset2/train/images/',
    'val':'D:/yolo_dataset2/valid/images/',
    'test':'D:/yolo_dataset2/test/images/',
    'nc':11,
    'names':['Vehicle_Car','Vehicle_Bus','Vehicle_Motorcycle','Vehicle_Unknown',
             'Pedestrian_Pedestrian','Pedestrian_Bicycle',
             'TrafficLight_Red','TrafficLight_Yellow','TrafficLight_Green',
             'RoadMark_StopLine','RoadMark_Crosswalk']
}

# yaml 파일 생성
with open('D:/yolo_dataset2/data.yaml', 'w') as f:
    yaml.dump(data, f)

# yaml 파일 읽기
with open('D:/yolo_dataset2/data.yaml', 'r') as f:
    vehicles_yaml = yaml.safe_load(f)

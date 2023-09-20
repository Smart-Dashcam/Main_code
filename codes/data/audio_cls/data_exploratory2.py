import os
import json
import pickle
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

new_folder = os.path.join('D:/','음성분류2')
# folder_path = "D:/위급상황 음성_음향"
# label_path, sound_path = [os.path.join(folder_path,x) for x in os.listdir(folder_path) if x[0]=='[']

# print(label_path, sound_path)
#############################################################################################
# TASK 1. 사용할 파일 선정하기(붕괴사고)
#############################################################################################
# 불러오기
with open('D:/음성분류/label_types.pkl','rb') as f:
    label_types = pickle.load(f)

with open('D:/음성분류/voice.pkl','rb') as f:
    voice = pickle.load(f)

# print(len(label_types))
print(label_types)
# print(len(voice))
print(voice)
# print(voice.keys())

mycond_types = ['낙석떨어지는소리', '낙석이떨어지는상황', '붕괴소리', '건물잔해가떨어지는상황', '창문이깨지는상황']
mycond_voice = ['대피해','붕괴됐어','붕괴됬어','대피하세요','대피하세요.','도로가 매몰됐어요','집이 무너졌어요','산사태가 났어','산사태가 났어요','무너졌다','다리가 무너졌어요 다리가']

for type_name in mycond_types:
    label_types[type_name] = 1
for voice_name in mycond_voice:
    voice[voice_name] = 1


if not os.path.isfile(new_folder + '/task1_complete.txt'):
    print('> Search File!', end=' ')
    use_files = []
    cnt = 0
    for jsonfile in os.listdir(label_path):
        json_path = os.path.join(label_path,jsonfile).replace('\\','/')

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_contents = json.load(f)

            temp = json_contents['annotations'][0]
            type_name = temp['categories']['category_03'].split('_')[-1]
            voice_name = temp['note']

            if label_types[type_name] == 1 and voice[voice_name] == 1:
                temp = jsonfile.replace('.json','_label.wav')
                use_files.append([os.path.join(sound_path,temp),0])

                if cnt % 200 == 0:
                    print(f'> {cnt}', end=' ')
                if cnt + 1 == 5000:
                    break
                cnt += 1

        except Exception as e:
            pass
    print('End')
    print(f'Total Count: {cnt}')

    useData = pd.DataFrame(use_files, columns=['file_path','label'])
    useData.to_csv(new_folder + f'/useData_collapse.csv',index=False)

    with open(new_folder + '/task1_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 2. 사용할 파일 선정하기(정상음)
#############################################################################################
if not os.path.isfile(new_folder + '/task2_complete.txt'):
    folder_path = os.path.join('D:','130.도시 소리 데이터')
    folder_list = ['3.차량주행음','5.이륜차주행음']

    data_normal = pd.DataFrame(columns=['file_path','label'])
    for folder in folder_list:
        temp = pd.DataFrame()
        filelist = os.listdir(os.path.join(folder_path,folder))
        temp['file_path'] = filelist
        temp['label'] = 1
        temp['file_path'] = temp['file_path'].apply(lambda x: os.path.join(folder_path,folder,x))

        data_normal = pd.concat([data_normal,temp], axis=0)


    data_normal.iloc[:5000].to_csv(os.path.join(new_folder,'useData_normal.csv'),index=False)

    with open(new_folder + '/task2_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 4. train, val, test 생성
#############################################################################################
if not os.path.isfile(new_folder + '/task3_complete.txt'):
    for folder in ['train','valid','test']:
        for sub_folder in ['주행음','충돌음']:
            folder_path = os.path.join(new_folder,folder,sub_folder)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
            print('pass')

    with open(new_folder + '/task3_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 5. train, val, test에 이미지 배분하기
#############################################################################################
# train, val, test 폴더 만들기
train_cnt = 0.7 * 4000
val_cnt = 0.1 * 4000
test_cnt = 0.2 * 4000

new_folder = os.path.join('D:','음성분류2')
for folder in tqdm(['주행음', '충돌음']):
    folder_path = os.path.join(new_folder, folder)
    cnt = 0
    for image in os.listdir(folder_path):
        if image[-3:] != 'jpg':
            continue
        curr_path = os.path.join(folder_path, image)
        if cnt < train_cnt:
            move_path = os.path.join(new_folder, 'train', folder, image)
        elif cnt < train_cnt + val_cnt:
            move_path = os.path.join(new_folder, 'valid', folder, image)
        else:
            move_path = os.path.join(new_folder, 'test', folder, image)

        copyfile(curr_path, move_path)
        cnt += 1



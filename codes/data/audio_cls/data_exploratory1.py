import os
import json
import pickle
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

new_folder = 'D:/음성분류'
folder_path = "D:/위급상황 음성_음향"
label_path, sound_path = [os.path.join(folder_path,x) for x in os.listdir(folder_path) if x[0]=='[']

print(label_path, sound_path)
#############################################################################################
# TASK 1. label type 파악하기
#############################################################################################
if not os.path.isfile(new_folder + '/task1_complete.txt'):
    label_types = set()
    voice = set()

    print('> Read Jsonfile!', end=' ')
    exclude_file = []
    for i, jsonfile in enumerate(os.listdir(label_path)):
        if i % 2000 == 0:
            print(f'> {i}', end=' ')

        json_path = os.path.join(label_path,jsonfile).replace('\\','/')

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_contents = json.load(f)

            temp = json_contents['annotations'][0]
            label_types.add(temp['categories']['category_03'].split('_')[-1])
            voice.add(temp['note'])
        except:
            exclude_file.append(jsonfile)
            pass
    print('End')
    print('except',exclude_file)

    with open(new_folder + '/label_types.pkl','wb') as f:
        pickle.dump(dict.fromkeys(list(label_types),0),f)

    with open(new_folder + '/voice.pkl','wb') as f:
        pickle.dump(dict.fromkeys(list(voice),0),f)

    with open(new_folder + '/task1_complete.txt','w') as f:
        f.write('complete')

# 불러오기
with open(new_folder + '/label_types.pkl','rb') as f:
    label_types = pickle.load(f)

with open(new_folder + '/voice.pkl','rb') as f:
    voice = pickle.load(f)

# print(len(label_types))
print(label_types)
# print(len(voice))
print(voice)
# print(voice.keys())

mycond_types = ['낙석떨어지는소리', '낙석이떨어지는상황', '붕괴소리', '건물잔해가떨어지는상황', '창문이깨지는상황']
mycond_voice = ['대피해','붕괴됐어','붕괴됬어','대피하세요','대피하세요.']

for type_name in mycond_types:
    label_types[type_name] = 1
for voice_name in mycond_voice:
    voice[voice_name] = 1

# print(label_types)
# print(voice)
#############################################################################################
# TASK 2. 사용할 파일 선정하기(붕괴사고)
#############################################################################################
if not os.path.isfile(new_folder + '/task2_complete.txt'):
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
                use_files.append([os.path.join(sound_path,temp).replace('\\','/'),0])

                if cnt % 200 == 0:
                    print(f'> {cnt}', end=' ')
                if cnt + 1 == 2500:
                    break
                cnt += 1

        except Exception as e:
            pass
    print('End')
    print(f'Total Count: {cnt}')

    useData = pd.DataFrame(use_files, columns=['file_name','label'])
    useData.to_csv(new_folder + f'/useData_collapse.csv',index=False)

    with open(new_folder + '/task2_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 3. 사용할 파일 선정하기(경적음)
#############################################################################################
if not os.path.isfile(new_folder + '/task3_complete.txt'):
    folder_path = 'D:/130.도시 소리 데이터'
    use_files_horn, use_files_normal = os.listdir(folder_path)

    for folder in os.listdir(folder_path):
        file_path = os.path.join(folder_path,folder)
        use_files = []
        label = 1 if '주행음' in folder else 0
        max_count = 5000 if '주행음' in folder else 2500
        for cnt, wavfile in enumerate(os.listdir(file_path)):
            wav_path = os.path.join(file_path,wavfile)
            use_files.append([wav_path.replace('\\','/'), label])

            if cnt + 1 == max_count:
                break

        save_name = 'normal' if '주행음' in folder else 'horn'
        useData = pd.DataFrame(use_files, columns=['file_name', 'label'])
        useData.to_csv(new_folder + f'/useData_{save_name}.csv',index=False)

    with open(new_folder + '/task3_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 4. train, val, test 생성
#############################################################################################
if not os.path.isfile(new_folder + '/task4_complete.txt'):
    for folder in ['train','valid','test']:
        for sub_folder in ['주행음','충돌음']:
            folder_path = os.path.join(new_folder,folder,sub_folder)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            print('pass')

    with open(new_folder + '/task4_complete.txt','w') as f:
        f.write('complete')

#############################################################################################
# TASK 5. train, val, test에 이미지 배분하기
#############################################################################################
# train, val, test 폴더 만들기
train_cnt = 0.7 * 5000
val_cnt = 0.1 * 5000
test_cnt = 0.2 * 5000

new_folder = 'D:/음성분류'
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



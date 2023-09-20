import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 데이터 병합
data1 = pd.read_csv('useData_normal.csv')
data2 = pd.read_csv('useData_collapse.csv')
data3 = pd.read_csv('useData_horn.csv')

data = pd.concat([data1,data2,data3],axis=0)

# 멜 스펙토그램 생성 및 저장
create_folder = 'D:/음성분류'

for i in tqdm(range(data.shape[0])):
    file_path, label = data.iloc[i, :]
    folder = '주행음' if label == 1 else '충돌음'
    file_name = file_path.split('/')[-1].replace('.wav', '')
    save_path = os.path.join(create_folder, folder, file_name)

    try:
        # 오디오 불러오기
        y, sr = librosa.load(file_path)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 이미지 그리기
        fig, ax = plt.subplots(figsize=(5, 4))
        librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=ax)
        ax.axis('off')
        # plt.show()

        # 이미지 저장하기
        plt.savefig(save_path + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(e)


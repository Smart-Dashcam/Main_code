import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import shutil
import matplotlib.pyplot as plt

def audio_preprocessing(file_path,label):
    label_dict = {1:'주행음',0:'충돌음'}

    file_name = file_path.split('\\')[-1]
    move_path = os.path.join('D:/음성분류2',label_dict[label],file_name)

    # 전처리 1 - 오디어 분리
    cmd_txt = f"spleeter separate -o output \"{file_path}\""
    os.system(cmd_txt)

    # 전처리 2 - 정적 소음 제거
    curr_path = 'C:/Users/user/metaverse/project8/audio'
    output_path = os.path.join(curr_path, 'output', file_name[:-4])
    shutil.move(os.path.join(output_path,'accompaniment.wav'), move_path)
    y, sr = librosa.load(move_path)

    y_trimmed, index = librosa.effects.trim(y, top_db=60)
    sf.write(move_path,y_trimmed,sr)

    # 파일 제거
    shutil.rmtree(os.path.join(output_path))

    # 오디오 불러오기
    y, sr = librosa.load(move_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 이미지 그리기
    fig, ax = plt.subplots(figsize=(5, 4))
    librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off')

    # 이미지 저장하기
    plt.savefig(move_path[:-3] + 'jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':

    # test
    # audio_preprocessing("samples/11.붕괴사고_479012_label.wav", 0)

    # 데이터 병합
    # data1 = pd.read_csv('D:/음성분류2/useData_normal.csv')
    # data2 = pd.read_csv('D:/음성분류2/useData_collapse.csv')

    # data = pd.concat([data1, data2], axis=0)
    data = pd.read_csv('data_test.csv')
    # print(data)
    print(data.shape)

    for i in range(data.shape[0]):
        file_path, label = data.iloc[i, :]
        folder = '주행음' if label == 1 else '충돌음'

        audio_preprocessing(file_path,label)
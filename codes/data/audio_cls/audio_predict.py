import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image

import speech_recognition as sr
import playsound
import time

# 주행음 1 충돌음 0
audiofile = './오토바이충돌.mp3'
def draw_melspectogram(audiofile):
    # 오디오 불러오기
    y, sr = librosa.load(audiofile)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 이미지 그리기
    fig, ax = plt.subplots(figsize=(5, 4))
    librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off')
    # plt.show()

    # 이미지 저장하기
    savefile = audiofile[:-4] + '.jpg'
    plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
    plt.close()

    return savefile

model = torch.load('./audio.pth',map_location=torch.device('cpu'))
def audio_predict(audiofile, model):
    file_path = draw_melspectogram(audiofile)
    image = Image.open(file_path)

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229,0.224,0.225))
        ]
    )
    image = test_transforms(image).unsqueeze(0).to('cpu')
    model.to('cpu')

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs.data,1)

    return pred.item()

if __name__ == '__main__':
    audio_pred = audio_predict(audiofile, model)
    print(audio_pred)

    curr_path = os.getcwd()
    if audio_pred == 0:
        r = sr.Recognizer()
        cnt = 0

        playsound.playsound(os.path.join(curr_path, 'tts', '01_사고를_감지했습니다_신고가_필요하신가요.mp3'))
        time.sleep(0.2)
        # print('>>> 사고가 감지되었습니다. 신고가 필요하신가요?')
        while True:
            # time.sleep(0.5)
            with sr.Microphone() as source:
                audio = r.listen(source, phrase_time_limit=1)
                try:
                    # 소리 인식
                    audio_text = r.recognize_google(audio, language='ko-KR')
                    print('[사용자] ' + audio_text)

                    # 사고가 났을 때
                    if '도와줘' in audio_text:
                        playsound.playsound(os.path.join(curr_path, 'tts', '03_등록된_지인에게_문자를_발송하겠습니다.mp3'))
                        time.sleep(0.2)
                        # print('등록된 지인에게 문자를 발송하겠습니다.')
                        break
                    # 사고가 나지 않았을 때
                    elif '아니' in audio_text:
                        playsound.playsound(os.path.join(curr_path ,'tts','02_알겠습니다_안전_운행하세요.mp3'))
                        time.sleep(0.2)
                        # print('네. 안전운행하세요.')
                        break
                except:
                    pass

                # 무응답일 때
                cnt += 1
                if cnt == 3:
                    playsound.playsound(os.path.join(curr_path, 'tts', '04_계속된_무응답으로_등록된_지인에게_문자를_발송하겠습니다.mp3'))
                    time.sleep(0.2)
                    # print('>>> 계속된 무응답으로 등록된 지인에게 문자를 발송하겠습니다.')
                    break

                # 조건에 해당되지 않으면 다시 물어본다.
                playsound.playsound(os.path.join(curr_path, 'tts', '05_다시_한번_말씀해주세요.mp3'))
                time.sleep(0.2)
                # print('>>> 다시 한번 말씀해주세요.')

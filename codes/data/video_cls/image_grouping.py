import os
import image2video

group_dict = {}
folder_path = 'D:/088.승용 자율주행차 주간 도심도로 데이터/01.데이터/Validation/01.원천데이터/'
save_path = 'D:/vivit/mydata/'
folders = os.listdir(folder_path)

for i, folder in enumerate(folders,1):
    if folder[-3:] == 'zip':
        continue
    subfolder_path = folder_path + folder
    print(subfolder_path)
    print(os.path.isdir(subfolder_path))
    subsave_path = os.path.join(save_path, f'driving_video_{i}.mp4')
    image2video.convert_frames_to_video(subfolder_path,subsave_path,fps=7)

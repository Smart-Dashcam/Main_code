import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    
    for i in range(len(files)):
        filename= os.path.join(pathIn,files[i])
        filename_arr = np.fromfile(filename, np.uint8)
        #reading each files
        img = cv2.imdecode(filename_arr, cv2.IMREAD_COLOR)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    pathIn= 'D:/088.승용 자율주행차 주간 도심도로 데이터/01.데이터/Validation/01.원천데이터/VS_08_085504_221103_sensor_raw_data_camera/'
    pathOut = 'D:/vivit/video.avi'
    fps = 7.0
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()


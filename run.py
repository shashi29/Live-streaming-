import cv2
import numpy as np
import sys
import os
import io
import pandas as pd
import glob

from collections import Counter 
from ffpyplayer.player import MediaPlayer

#Google API
import wave
from typing import Tuple
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import vision

#Threading
import threading
import time

import math
import time
import copy

from utility import *
import math
import wave
import struct
import numpy as np
import pandas as pd


from pydub import AudioSegment
from pydub.playback import play
import numpy as np
from scipy.io import wavfile
#Read from pickel file info
import pickle

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="credentials.json"

def detect_text_googleVisonApi(path, frame_count):
    with open('mask_word.txt') as fp1: 
        mask_word = fp1.read() 
            
    mask_word = mask_word.split("\n")
    client = vision.ImageAnnotatorClient()

    is_success, im_buf_arr = cv2.imencode(".jpg", path)
    content = im_buf_arr.tobytes()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    file_name = f'intermediate/temp_{frame_count}.txt'
    file = open(file_name,'w')

    for text in texts:
        if text.description in mask_word:            
            x_list = list()
            y_list = list()
            for vertex in text.bounding_poly.vertices:
                x_list.append(vertex.x)
                y_list.append(vertex.y)
            xmin = min(x_list)
            ymin = min(y_list)
            xmax = max(x_list)
            ymax = max(y_list)
            
            file.write(f'{xmin} {ymin} {xmax} {ymax}\n')

    file.close()
    fp1.close()
    print(f"[INFO] Processing Frame {frame_count}")


def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def find_text_region_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    crop_img = img[y:y+h,x:x+w]
    return [x,y,x+w,y+h],crop_img


def extract_mask_bbox_info(video_path): 
    video=cv2.VideoCapture(video_path)
    count = 0
    crop_list = list()
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        if not grabbed:
            print("[INFO] End of video")
            break
        
        height, width, channel = frame.shape
        lower_height = int(6/8*height)
        img = frame[lower_height:height,0:width]
        
            
        if len(crop_list) > 1:
            prev_frame = crop_list[-1]
            prev_img = prev_frame[lower_height:height,0:width]
            boxA, crop_imgA = find_text_region_crop(img)
            boxB, crop_imgB = find_text_region_crop(prev_img)            
            iou = bb_intersection_over_union(boxA, boxB)
            if iou < 1.0:
                th = threading.Thread(target = detect_text_googleVisonApi, args=(img, count, ))
                th.daemon = True
                th.start()
        crop_list.append(frame)


def playVideo(video_path):
    video=cv2.VideoCapture(video_path)
    count = -1
    previous_processed_frames = list()
    previous_processed_frames.append(0)     #Hack for first frame
    while (video.isOpened()):
        count = count + 1
        grabbed, frame=video.read()
        if not grabbed:
            print("[INFO] End of video")
            break
                    
        height, width, channel = frame.shape
        Offset = int(6/8*height)
        #print(f"[INFO] Frame {count}")
        file_name = f'intermediate/temp_{count}.txt'
        if os.path.isfile(file_name):
            previous_processed_frames.append(count)
            #print(f"[INFO] Processing Frames {count}")
            with open(file_name) as fp1: 
                mask_word_box = fp1.read() 
            
            mask_word_box = mask_word_box.split("\n")
            mask_word_box.pop()
            if len(mask_word_box) > 0:
                for bb_mask in mask_word_box:
                    bb_mask = bb_mask.split(" ")
                    cv2.rectangle(frame,(int(bb_mask[0]),int(bb_mask[1])+Offset),(int(bb_mask[2]),int(bb_mask[3])+Offset),(0,0,255),-1)
        fp1.close()

        if os.path.isfile(file_name) == False:
            prev_count = previous_processed_frames[len(previous_processed_frames)-1]
            file_name = f"intermediate/temp_{prev_count}.txt"
            if os.path.isfile(file_name):
                with open(file_name) as fp1: 
                    mask_word_box = fp1.read() 
                
                mask_word_box = mask_word_box.split("\n")
                mask_word_box.pop()
                if len(mask_word_box) > 0:
                    for bb_mask in mask_word_box:
                        bb_mask = bb_mask.split(" ")
                        cv2.rectangle(frame,(int(bb_mask[0]),int(bb_mask[1])+Offset),(int(bb_mask[2]),int(bb_mask[3])+Offset),(0,0,255),-1)
            fp1.close()

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        cv2.imshow("Video", frame)
    video.release()
    cv2.destroyAllWindows()

def word_timestamp(response):
    word_list = list()
    word_start_sec_list = list() 
    word_start_nano_sec_list = list()
    word_end_sec_list = list()
    word_end_nano_sec_list = list()
    word_start_time_list = list()
    word_end_time_list = list()

    for result in response.results:
        for i in range(len(result.alternatives[0].words) - 1):
            word = result.alternatives[0].words[i].word
            word = word.lower()
            #if word in bad_word:
            word_start_sec = result.alternatives[0].words[i].start_time.seconds
            word_end_sec = result.alternatives[0].words[i].end_time.seconds
            word_start_nano_sec = result.alternatives[0].words[i].start_time.nanos
            word_end_nano_sec = result.alternatives[0].words[i].end_time.nanos
            word_start_time = word_start_sec * pow(10,3) + word_start_nano_sec * pow(10,-6)
            word_end_time = word_end_sec * pow(10,3) + word_end_nano_sec * pow(10,-6)
            word_list.append(word)
            word_start_sec_list.append(word_start_sec)
            word_start_nano_sec_list.append(word_start_nano_sec)
            word_end_sec_list.append(word_end_sec)
            word_end_nano_sec_list.append(word_end_nano_sec)
            word_start_time_list.append(word_start_time)
            word_end_time_list.append(word_end_time)

    #Make starting point to zer0
    word_start_time_lag = copy.deepcopy(word_start_time_list)
    del word_start_time_lag[0]
    word_start_time_lag.append(-1)
    word_start_time_list[0] = 0
    
    df = pd.DataFrame()
    df['word'] = word_list
    df['word_start_sec'] = word_start_sec_list 
    df['word_start_nano_sec'] = word_start_nano_sec_list
    df['word_end_sec'] = word_end_sec_list
    df['word_end_nano_sec'] = word_end_nano_sec_list
    df['word_start_time'] = word_start_time_list
    df['word_end_time'] = word_end_time_list
    df['word_start_time_lag'] = word_start_time_lag

    return df    

def create_mask_audio(word_duration, beep_audio):
    if len(beep_audio) >= word_duration:
        return beep_audio[:word_duration]
    else:
        new_beep_audio = beep_audio * (int(word_duration//len(beep_audio))+1)
        return new_beep_audio[:word_duration]
    
def process_audio(audio_path, beep_path, df, bad_word):
    audio = AudioSegment.from_wav(audio_path)
    beep_audio = AudioSegment.from_wav(beep_path)
    # with open('mask_word.txt') as fp1: 
    #     mask_word = fp1.read() 

    # bad_word = mask_word.split("\n")
    mask_audio = AudioSegment.empty()
    for index, row in df.iterrows():
        try:
            word = row['word']
            word_start_time = row['word_start_time']
            word_end_time = row['word_start_time_lag']
            word_duration = word_end_time - word_start_time
            if word in bad_word:
                mask_audio_word = create_mask_audio(word_duration,beep_audio) 
                #mask_audio_word = AudioSegment.silent(duration = word_duration)
                mask_audio += mask_audio_word
            else:
                mask_audio += audio[word_start_time:word_end_time]
    
        except Exception as ex:
            print(f"[ERROR] {ex}")
            continue
        
    return mask_audio

if __name__ == "__main__":

    video_path = "5.mp4"
    audio_path = "audio.wav"
    beep_path = "beep.wav"
    BUCKET_NAME = "audio_2020"

    os.remove(audio_path)        
    #Remove intermediate files
    for temp_files in glob.glob("intermediate/*"):
        os.remove(temp_files)
    
    th = threading.Thread(target = extract_mask_bbox_info, args=(video_path, ))
    th.daemon = True
    th.start()
    print("[INFO] Find details of all mask word in video")
    file = open('intermediate/temp_0.txt','w')
    file.close()
        
    channels, bit_rate, sample_rate = video_info(video_path)
    blob_name = video_to_audio(video_path, audio_path, channels, bit_rate, sample_rate)

    gcs_uri = f"gs://{BUCKET_NAME}/{audio_path}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)
    

    with open('entry.pickle', 'wb') as f:
        pickle.dump(response, f)
    response = pickle.load(open( "entry.pickle", "rb" ))
    
    response_df = word_timestamp(response)
    
    #mask audio
    mask_audio = process_audio(audio_path, beep_path, response_df)
    print(mask_audio)
    mask_audio.export("final.wav", format="wav")

    print("[INFO] Starting video")
    startDemo = threading.Thread(target = playVideo, args=(video_path, ))
    startDemo.daemon = True
    startDemo.start()
    
    import simpleaudio as sa

    # Input an existing wav file name
    wavFile = "final.wav"
     
    # Play the sound if the wav file exists
    try:
        # Define object to play
        w_object = sa.WaveObject.from_wave_file(wavFile)
        # Define object to control the play
        p_object = w_object.play()
        print("Sound is playing...")
        p_object.wait_done()
        print("Finished.")
    
    # Print error message if the file does not exist
    except FileNotFoundError:
        print("Wav File does not exists")
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import os
import uvicorn
import json
import string
from collections import Counter
from utility import *
from run import *
from pydantic import BaseModel
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from process_video import *

app = FastAPI()
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
ALLOWED_EXTENSIONS = {'mp4'}
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def delete_wav_file():
    for filename in glob.glob('*.wav'):
        if filename != 'beep.wav':
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
    for p in Path().glob('*.wav'):
        if p.name != 'beep.wav':
            try:
                p.unlink()
            except Exception as e:
                print(f"Error deleting {p}: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_ngrams(transcript: str, n: int):
    """
    Counts the n-grams (unigrams, bigrams, trigrams) in the given transcript.

    Args:
        transcript (str): The transcript text.
        n (int): The order of n-grams to count.

    Returns:
        Counter: A Counter object containing the count of each n-gram.
    """
    # Remove punctuation and convert to lowercase
    transcript = transcript.translate(str.maketrans('', '', string.punctuation)).lower()

    # Split the transcript into words
    words = transcript.split()

    # Count the n-grams using the sliding window technique
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)

    return ngram_counts


def convert_tuple_to_dict(output):
    converted_output = {}

    for words, count in output.items():
        key = ' '.join(words)
        converted_output[key] = count

    return converted_output

def generate_output_json(transcript: str, unigram_counts: Counter, bigram_counts: Counter, trigram_counts: Counter) -> str:
    """
    Generates the output JSON structure containing the transcript and n-gram counts.

    Args:
        transcript (str): The transcript text.
        unigram_counts (Counter): The count of unigrams.
        bigram_counts (Counter): The count of bigrams.
        trigram_counts (Counter): The count of trigrams.

    Returns:
        str: The output JSON string.
    """
    output = {
        'transcript': transcript,
        'unigram_counts': convert_tuple_to_dict(dict(unigram_counts)),
        'bigram_counts': convert_tuple_to_dict(dict(bigram_counts)),
        'trigram_counts': convert_tuple_to_dict(dict(trigram_counts))
    }
    output = {str(key): value for key, value in output.items()}

    output_json = json.dumps(output, indent=4)
    return output_json

def process_video(video_path, filename):
    video_name = os.path.basename(video_path)
    video_name = video_name.split(".")[0]
    raw_audio_name = f'{video_name}_audio.wav'
    raw_audio_path = raw_audio_name
    BUCKET_NAME = "audio_2020"

    channels, bit_rate, sample_rate = video_info(video_path)
    blob_name = video_to_audio(video_path, raw_audio_path, channels, bit_rate, sample_rate)
    
    gcs_uri = f"gs://{BUCKET_NAME}/{raw_audio_name}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)
    # Accumulate the transcript text
    transcript = ''
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        transcript += result.alternatives[0].transcript + ' '
    
    # Count the n-grams
    unigram_counts = count_ngrams(transcript, 1)
    bigram_counts = count_ngrams(transcript, 2)
    trigram_counts = count_ngrams(transcript, 3)

    # Generate the output JSON
    output_json = generate_output_json(transcript, unigram_counts, bigram_counts, trigram_counts)
    delete_wav_file()
    return output_json

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    delete_wav_file()
    filename = file.filename
    
    if not allowed_file(filename):
        return {"error": "Invalid file type"}
        
    file_location = f"{UPLOAD_FOLDER}/{filename}"
    
    # Save uploaded file to disk
    with open(file_location, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video
    try:
        output = process_video(file_location, filename)
    except Exception as e:
        return {"error": str(e)}

    return output

class MuteRequest(BaseModel):
    video_name: str
    words_to_mute: List[str]
    enable_subititle_mask: int

def mask_video_subtitile(input_video_path, no_audio_video_path, WORDS_TO_MUTE):
    interval = 1
    # Queues for frames and detected text
    text_queue = list()
    frame_queue = list()
    frame_queue = extract_frames(input_video_path, interval, frame_queue)
    for frame_info in frame_queue:
        frame = frame_info[0]
        frame_index = frame_info[1]
        if frame_index % 10 == 0:
            masked_words_box, frame_index, masked_words = process_frame(frame, frame_index, WORDS_TO_MUTE)
            info = dict()
            info['frame_index'] = int(frame_index)
            info['bounding_box'] = masked_words_box if len(masked_words_box) else []
            info['mask_word'] = masked_words
            text_queue.append(info)
        else:
            info = dict()
            info['frame_index'] = int(frame_index)
            info['bounding_box'] =  []
            info['mask_word'] = ''
            text_queue.append(info)

    #text_queue = process_frames_in_parallel(frame_queue, WORDS_TO_MUTE)
    text_queue = pd.DataFrame(text_queue)
    #text_queue.to_csv("out4_test.csv", index=False)
    threshold = 2
    frame_ranges = find_similar_frame_ranges(input_video_path, threshold)
    text_queue = fill_missing_frames(frame_ranges, text_queue)

    # Call the function with the sample data
    #text_queue = fill_empty_rows(text_queue)
    save_processed_video(input_video_path, no_audio_video_path, text_queue)

@app.post("/mute_video")
async def mute_video(request: MuteRequest):
    delete_wav_file()
    video_name = request.video_name
    words_to_mute = request.words_to_mute
    
    print("Words to mute", words_to_mute)
    video_path = os.path.join(os.getcwd(), video_name)
    video_name = os.path.basename(video_path)
    video_name = video_name.split(".")[0]
    raw_audio_name = f'{video_name}_audio.wav'
    beep_path = "beep.wav"
    raw_audio_path = os.path.join(os.getcwd(), raw_audio_name)
    processed_audio_name = f'{video_name}_final.wav'
    processed_audio_path = os.path.join(os.getcwd(), processed_audio_name)
    BUCKET_NAME = "audio_2020"
    no_audio_video_path = video_path[:-4] + '_No_Audio.mp4'
    filename = video_path[:-4] + '_final.mp4'
    processed_video = os.path.join(os.getcwd(),filename)
    
    channels, bit_rate, sample_rate = video_info(video_path)
    blob_name = video_to_audio(video_path, raw_audio_path, channels, bit_rate, sample_rate)
    
    gcs_uri = f"gs://{BUCKET_NAME}/{raw_audio_name}"
    response = long_running_recognize(gcs_uri, channels, sample_rate)
    response_df = word_timestamp(response)

    words_to_mute = [word.lower() for string in words_to_mute for word in string.split()]

    #mask audio
    mask_audio = process_audio(raw_audio_path, beep_path, response_df, words_to_mute)
    mask_audio.export(processed_audio_path, format="wav")

    #Enable subtitle mask
    if request.enable_subititle_mask:
        mask_video_subtitile(video_path, no_audio_video_path, words_to_mute)
    else:
        #Remove audio 
        command = f"ffmpeg -i {video_path} -vcodec copy -an -y {no_audio_video_path}"
        os.system(command)
    
    command = f"ffmpeg -i {no_audio_video_path} -i {processed_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -y {processed_video}"
    os.system(command)
    #add_srt_video("subtitles.srt", processed_video)
    delete_wav_file()
    return FileResponse(path=processed_video, media_type='video/mp4')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

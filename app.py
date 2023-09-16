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


app = FastAPI()
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
ALLOWED_EXTENSIONS = {'mp4'}
origins = ["*"] 

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


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

    # words_to_mute = wordlist.words
    #words_to_mute = ["machine learning", "to", "the"]
    words_to_mute = [word.lower() for string in words_to_mute for word in string.split()]

    #mask audio
    mask_audio = process_audio(raw_audio_path, beep_path, response_df, words_to_mute)
    mask_audio.export(processed_audio_path, format="wav")
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
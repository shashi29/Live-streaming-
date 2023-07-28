import subprocess
import os
import logging
import wave
import string
import json
from typing import Tuple, List
from datetime import datetime
from google.cloud import storage, speech_v1p1beta1 as speech
from google.cloud.speech import SpeechClient, RecognitionAudio, RecognitionConfig
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
from fastapi import FastAPI, Request

from pydub import AudioSegment
from pydub.silence import split_on_silence
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from urllib.parse import urlparse
from pydub.utils import mediainfo


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspaces/Live-streaming-/backend/google_env.json"

app = FastAPI()


logging.basicConfig(level=logging.INFO)


class VideoProcessingRequest(BaseModel):
    gcs_bucket_path: str

class InputPayload(BaseModel):
    gcs_bucket_path: str

def parse_gcs_uri(gcs_uri: str) -> tuple:
    """Parses the GCS URI and returns the bucket name and object name."""
    parts = gcs_uri.replace("gs://", "").split("/")
    bucket_name = parts[0]
    object_name = "/".join(parts[1:])
    return bucket_name, object_name

def download_video(gcs_uri: str, local_path: str) -> None:
    """Downloads the video file from the Google Cloud Storage (GCS) bucket to the local system."""
    logging.info(f"Downloading video from {gcs_uri} to {local_path}...")
    client = storage.Client()
    bucket_name, object_name = parse_gcs_uri(gcs_uri)

    # Download the file from GCS to the local system
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.download_to_filename(local_path)

    logging.info("Video downloaded successfully.")


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extracts audio from a video file using ffmpeg."""
    logging.info(f"Extracting audio from {video_path} to {audio_path}...")
    # Create audio folder if not exists
    audio_folder = 'audio'
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    # Extract audio from video using ffmpeg
    command = f"ffmpeg -i {video_path} -acodec pcm_s16le -ac 1 -ar 16000 {audio_path}"
    subprocess.run(command.split(), check=True)

    logging.info(f"Audio extracted successfully.")


def upload_audio_to_gcs_bucket(bucket_name: str, audio_path: str) -> str:
    """Uploads the audio file to the given Google Cloud Storage (GCS) bucket.
    Returns the GCS path of the uploaded file.
    """
    logging.info(f"Uploading {audio_path} to GCS bucket {bucket_name}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create a folder with the current date in the bucket
    folder_name = datetime.now().strftime("%Y-%m-%d")
    folder_blob = bucket.blob(folder_name + '/')
    folder_blob.upload_from_string('')

    # Create an object name based on the file name
    object_name = folder_name + '/' + os.path.basename(audio_path)

    # Upload the file to the bucket
    blob = bucket.blob(object_name)
    blob.upload_from_filename(audio_path)

    # Return the GCS path of the uploaded file
    gcs_path = f'gs://{bucket_name}/{object_name}'
    logging.info(f"Audio uploaded to {gcs_path}")
    return gcs_path


def transcribe_audio(gcs_uri: str, sample_rate: int, channels: any) -> str:
    """
    Transcribes the audio file specified by the GCS URI using Google Cloud Speech-to-Text API.

    Args:
        gcs_uri (str): The GCS URI of the audio file.
        sample_rate (int): The sample rate of the audio file.

    Returns:
        str: The transcription result.

    Raises:
        Exception: If there is an error in the transcription process.
    """
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    logging.info(f"Transcribing audio from {gcs_uri}...")
    client = SpeechClient()

    audio = RecognitionAudio(uri=gcs_uri)
    config = {
        "language_code": "en-US",
        "sample_rate_hertz": int(sample_rate),
        "encoding": enums.RecognitionConfig.AudioEncoding.LINEAR16,
        "audio_channel_count": int(channels),
        "enable_word_time_offsets": True,
        "model": "video",
        "enable_automatic_punctuation":True
    }
    try:
        operation = client.long_running_recognize(config=config, audio=audio)

        logging.info("Waiting for operation to complete...")
        response = operation.result(timeout=90)

        # Accumulate the transcript text
        transcript = ''
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            transcript += result.alternatives[0].transcript + ' '

        logging.info(f"Transcription complete. Total text: {transcript}")
        return transcript, response

    except Exception as e:
        logging.error(f"Error occurred during transcription: {str(e)}")
        raise Exception("Transcription failed.")


def count_ngrams(transcript: str, n: int) -> Counter:
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


def get_audio_sample_rate(filename):
    """Reads a wav file and returns the audio data and sample rate."""
    logging.info(f"Reading {filename}...")
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    logging.info(f"{filename} read successfully")
    return buffer, rate


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File deleted successfully: {file_path}")
    except OSError as e:
        print(f"Error occurred while deleting file: {e}")

def video_info(video_filepath):
    """ this function returns number of channels, bit rate, and sample rate of the video"""

    video_data = mediainfo(video_filepath)
    channels = video_data["channels"]
    bit_rate = video_data["bit_rate"]
    sample_rate = video_data["sample_rate"]

    return channels, bit_rate, sample_rate

@app.get("/health")
async def health_check():
    return {"status": "App is running"}

@app.post("/process_video")
async def process_video(request: Request, video_request: VideoProcessingRequest):
    """
    Process the video file specified in the input payload.

    Args:
        request (Request): The FastAPI request object.
        video_request (VideoProcessingRequest): The video processing request containing the GCS bucket path.

    Returns:
        JSONResponse: The API response containing the output JSON structure.
    """
    try:
        # Extract the file name from the GCS bucket path
        file_name = os.path.basename(urlparse(video_request.gcs_bucket_path).path)

        # Define the local path to save the video file
        local_video_path = os.path.join("/workspaces/Live-streaming-/Code", file_name)

        # Download the video file
        download_video(video_request.gcs_bucket_path, local_video_path)

        file_name = file_name.split(".")[0] + ".wav"
        local_audio_path = os.path.join("/workspaces/Live-streaming-/Code", file_name)
        extract_audio(local_video_path, local_audio_path)

        # Upload the audio file to GCS
        gcs_uri = upload_audio_to_gcs_bucket("live-stream-video", local_audio_path)

        # Transcribe the audio
        #buffer, rate = get_audio_sample_rate(local_audio_path)
        channels, bit_rate, sample_rate = video_info(local_video_path)
        transcript, response = transcribe_audio(gcs_uri, bit_rate, channels)

        # Count the n-grams
        unigram_counts = count_ngrams(transcript, 1)
        bigram_counts = count_ngrams(transcript, 2)
        trigram_counts = count_ngrams(transcript, 3)

        # Generate the output JSON
        output_json = generate_output_json(transcript, unigram_counts, bigram_counts, trigram_counts)

        # Delete the temporary files
        delete_file(local_video_path)
        delete_file(local_audio_path)

        return JSONResponse(content=output_json, media_type="application/json")

    except Exception as e:
        logging.error(f"Error occurred during video processing: {str(e)}")
        return JSONResponse(content={"error": "Video processing failed."}, status_code=500)
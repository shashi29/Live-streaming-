import subprocess
import os
import logging
import wave
import string
from typing import Tuple
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_env.json"
from google.cloud import storage
from google.cloud.speech import SpeechClient, RecognitionAudio, RecognitionConfig
from collections import Counter
import os
from typing import List
from google.cloud import storage, speech_v1p1beta1 as speech
from pydub import AudioSegment
from pydub.silence import split_on_silence
from collections import Counter
import string
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.concatenate import concatenate_videoclips


logging.basicConfig(level=logging.INFO)

def upload_file_to_gcs_bucket(bucket_name: str, file_path: str) -> str:
    """Uploads a file to the given Google Cloud Storage bucket.
    Returns the GCS path of the uploaded file.
    """
    logging.info(f"Uploading {file_path} to GCS bucket {bucket_name}...")
    # Create a client object for the bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create an object name based on the file path
    object_name = '/'.join(file_path.split('/')[-2:])

    # Upload the file to the bucket
    blob = bucket.blob(object_name)
    blob.upload_from_filename(file_path)

    # Return the GCS path of the uploaded file
    gcs_path = f'gs://{bucket_name}/{object_name}'
    logging.info(f"File uploaded to {gcs_path}")
    return gcs_path


def extract_audio(video_path: str) -> str:
    """Extracts audio from a video file using ffmpeg"""
    logging.info(f"Extracting audio from {video_path}...")
    # create audio folder if not exists
    audio_folder = 'audio'
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    # extract audio from video using ffmpeg
    audio_file_path = os.path.join(audio_folder, os.path.splitext(os.path.basename(video_path))[0] + '.wav')
    command = f"ffmpeg -i {video_path} -acodec pcm_s16le -ac 1 -ar 16000 {audio_file_path}"
    subprocess.run(command.split(), check=True)

    logging.info(f"Audio extracted to {audio_file_path}")
    return audio_file_path


def transcribe_gcs(gcs_uri: str, rate: int) -> str:
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    logging.info(f"Transcribing audio from {gcs_uri}...")
    client = SpeechClient()

    audio = RecognitionAudio(uri=gcs_uri)
    config = RecognitionConfig(
        encoding="LINEAR16",
        sample_rate_hertz=rate,
        language_code="en-US",
    )

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


def read_wav_file(filename: str) -> Tuple[bytes, int]:
    """Reads a wav file and returns the audio data and sample rate."""
    logging.info(f"Reading {filename}...")
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    logging.info(f"{filename} read successfully")
    return buffer, rate

def get_word_frequency(transcript):
    words_frequency = Counter()

    # Remove punctuation and convert to lowercase
    transcript = transcript.translate(str.maketrans('', '', string.punctuation)).lower()

    # Count the frequency of each word in the transcript
    words = transcript.split()
    words_frequency.update(words)
    
    return words_frequency



def create_audio_with_blacklistwords(blacklist_words: List[str], response: speech.LongRunningRecognizeResponse, audio_path: str) -> str:
    """
    Creates a new audio file with blacklist words muted and returns the updated audio path.

    Args:
        blacklist_words (List[str]): A list of words to mute in the audio file.
        response (speech.LongRunningRecognizeResponse): The Google Speech-to-Text API response object.
        audio_path (str): The path to the audio file.

    Returns:
        str: The path to the updated audio file.
    """

    # Load audio file using PyDub
    audio = AudioSegment.from_file(audio_path)

    # Find the start and end time of each occurrence of blacklist words
    start_end_times = []
    for word in blacklist_words:
        for result in response.results:
            for alt in result.alternatives:
                if word.lower() in alt.transcript.lower() and alt.words:
                    start_time = alt.words[0].start_time.seconds + alt.words[0].start_time.nanos * 1e-9
                    end_time = alt.words[-1].end_time.seconds + alt.words[-1].end_time.nanos * 1e-9
                    start_end_times.append((start_time, end_time))

    # Mute audio during the occurrence of blacklist words
    for start, end in start_end_times:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        audio = audio.overlay(AudioSegment.silent(duration=end_ms-start_ms), position=start_ms)

    # Export the updated audio file
    basename = os.path.basename(audio_path)
    updated_audio_path = os.path.join(os.path.dirname(audio_path), "updated_" + basename)
    audio.export(updated_audio_path, format="wav")

    # Return the updated audio path
    return updated_audio_path

import os
import moviepy.editor as mp

def save_updated_video(original_video_path: str, updated_audio_path: str, output_dir: str, prefix: str) -> str:
    """
    Merges the original video file with the updated audio file and saves the result as a new video file.

    Args:
        original_video_path (str): Path to the original video file.
        updated_audio_path (str): Path to the updated audio file.
        output_dir (str): Directory to save the new video file in.
        prefix (str): Prefix to add to the name of the new video file.

    Returns:
        str: Path to the new video file.

    """
    # Load the original video file
    video_clip = mp.VideoFileClip(original_video_path)

    # Load the updated audio file
    audio_clip = mp.AudioFileClip(updated_audio_path)

    # Set the audio clip duration to match the video clip duration
    audio_clip = audio_clip.set_duration(video_clip.duration)

    # Combine the video clip and audio clip
    new_video_clip = video_clip.set_audio(audio_clip)

    # Save the new video file with the updated audio
    video_file_name = os.path.basename(original_video_path)
    video_file_name_without_extension = os.path.splitext(video_file_name)[0]
    new_video_path = os.path.join(output_dir, f"{prefix}_{video_file_name_without_extension}.mp4")
    new_video_clip.write_videofile(new_video_path, codec='libx264')

    # Close the clips
    video_clip.close()
    audio_clip.close()

    return new_video_path


audio_path = extract_audio('1.mp4')
gcs_url = upload_file_to_gcs_bucket("live-stream-video", audio_path)
buffer, rate = read_wav_file(audio_path)
transcript, response = transcribe_gcs(gcs_url, rate)
words_frequency = get_word_frequency(transcript)
print(words_frequency)
black_list_words = ['to', 'and', 'the', 'machine', 'you', 'a', 'that', 'is','learning']
new_audio_path = create_audio_with_blacklistwords(black_list_words, response, audio_path)
prefix = "updated"
output_dir = os.getcwd()
new_video_path = save_updated_video('1.mp4', new_audio_path, output_dir,  prefix)

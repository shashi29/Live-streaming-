import os
import re
import subprocess
from collections import Counter
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.cloud import storage
from moviepy.editor import VideoFileClip

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_env.json"

def transcribe_video_file(video_file_path, bucket_name, project_id):
    """Transcribes the audio from a video file using Google Speech-to-Text API."""
    client = speech_v1.SpeechClient()
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    # Upload the video file to Google Cloud Storage
    video_file_name = os.path.basename(video_file_path)
    blob = bucket.blob(video_file_name)
    blob.upload_from_filename(video_file_path)

    # Transcribe the audio from the video
    audio_uri = f"gs://{bucket_name}/{video_file_name}"
    audio = speech_v1.RecognitionAudio(uri=audio_uri)

    config = speech_v1.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US')

    response = client.recognize(config=config, audio=audio)

    # Parse the transcription results
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript

    return transcription, response

def get_word_frequency(transcription):
    """Returns the frequency of each word in the transcription."""
    # Remove all non-word characters and convert to lowercase
    transcription_cleaned = re.sub(r'\W+', ' ', transcription).lower()
    # Split the transcription into words
    words = transcription_cleaned.split()
    # Count the frequency of each word using the Counter class
    word_counts = Counter(words)
    # Return the dictionary of word frequencies
    return dict(word_counts)

def process_video(video_file_path, response, words_to_mute):
    """Mutes the audio for specific words in the video."""
    # Extract the words and their timestamps from the response
    words_info = response.results[0].alternatives[0].words

    # Get the duration of the video
    duration = VideoFileClip(video_file_path).duration

    # Mute the audio for each word
    video_file_name = os.path.basename(video_file_path)
    muted_video_file_path = f"{os.path.splitext(video_file_name)[0]}_muted.mp4"
    video = VideoFileClip(video_file_path)
    for word_info in words_info:
        word = word_info.word
        start_time = word_info.start_time.seconds + word_info.start_time.nanos / 1e9
        end_time = word_info.end_time.seconds + word_info.end_time.nanos / 1e9
        if word in words_to_mute:
            video = video.set_audio(video.audio.set_volume_at_time(0, start_time).set_volume_at_time(1, end_time))
    video.write_videofile(muted_video_file_path, audio_codec='aac')
    return muted_video_file_path

def stream_local_video(video_path, stream_url):
    """Streams a local video file using FFmpeg."""
    ffmpeg_command = f"ffmpeg -i {video_path} -c copy -f flv {stream_url}"
    subprocess.Popen(ffmpeg_command.split(), stdout=subprocess.PIPE).wait()

if __name__ == "__main__":
    # Set up GCP client and variables
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_env.json"
    bucket_name = "video_289315"
    project_id = "abiding-lead-289315"
    video_file_path = "1.mp4"
    bad_word_list = []

    # Transcribe the video and get word frequency
    transcription, response = transcribe_video_file(video_file_path, bucket_name, project_id)
    word_frequency = get_word_frequency(transcription)
    
    # Filter out bad words
    # for word, count in word_frequency.items():
    #     if is_bad_word(word):
    #         bad_word_list.append(word)

    # Process the video to mute bad words and stream it locally
    mute_video_file_path = process_video(video_file_path, response, bad_word_list)
    stream_local_video(mute_video_file_path, "rtmp://localhost/live/streamkey")

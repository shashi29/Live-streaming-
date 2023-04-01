import os
import io
import subprocess
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"


@app.route("/process-video", methods=["POST"])
def process_video():
    # Check if a file was uploaded in the request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the file from the request
    file = request.files["file"]

    # Check if the file is a video
    if not file.filename.endswith(('.mp4', '.mov', '.avi')):
        return jsonify({"error": "Invalid file format"}), 400

    # Save the file to disk
    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract audio from the video
    audio_file = os.path.join(app.config["UPLOAD_FOLDER"], f"{os.path.splitext(filename)[0]}.wav")
    subprocess.call(['ffmpeg', '-i', filepath, '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', audio_file])

    # Use Google Speech to Text API to transcribe the audio
    client = speech.SpeechClient()
    with io.open(audio_file, "rb") as audio:
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
        )
        audio = speech.RecognitionAudio(content=audio.read())
        try:
            response = client.recognize(config=config, audio=audio)
        except:
            return jsonify({"error": "Failed to transcribe audio"}), 500

    # Get the transcribed text from the response
    transcribed_text = ""
    for result in response.results:
        transcribed_text += result.alternatives[0].transcript.lower()

    # Remove punctuation and special characters
    transcribed_text = re.sub(r'[^\w\s]', '', transcribed_text)

    # Split the text into words
    words = transcribed_text.split()

    # Count the frequency of each word
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    # Sort the words by frequency
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # Return the unique words with their count
    unique_words = []
    for word, count in sorted_words:
        if count == 1:
            break
        unique_words.append({"word": word, "count": count})

    return jsonify({"unique_words": unique_words}), 200

def get_bad_word_counts(text, bad_words):
    """
    Count the occurrences of bad words in the given text.

    Parameters:
        text (str): The text to analyze.
        bad_words (list): A list of bad words to search for.

    Returns:
        dict: A dictionary where the keys are the bad words and the values are the counts of how many times each bad word appears in the text.
    """
    # Initialize the counts for each bad word to zero
    counts = {word: 0 for word in bad_words}

    # Split the text into words and iterate over each word
    for word in text.split():
        # If the word is in the list of bad words, increment its count
        if word in bad_words:
            counts[word] += 1

    return counts

@app.route("/mute-bad-words", methods=["POST"])
def mute_bad_words():
    # get the list of bad words entered by the user
    bad_words = request.json["bad_words"]
    
    # get the uploaded video file
    video_file = request.files["video"]
    video_filename = secure_filename(video_file.filename)
    
    # save the video file to disk
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    video_file.save(video_path)
    
    # extract the audio from the video
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "audio.wav")
    command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}"
    subprocess.call(command, shell=True)
    
    # transcribe the audio using Google Cloud Speech-to-Text API
    client = speech.SpeechClient()
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    
    # get the list of unique bad words and their counts
    bad_word_counts = get_bad_word_counts(response, bad_words)
    
    # mute the audio segments containing the bad words
    audio = AudioSegment.from_wav(audio_path)
    for bad_word in bad_words:
        if bad_word in bad_word_counts:
            # get the start and end times of the bad word segment
            for segment in bad_word_counts[bad_word]:
                start_time = segment[0] * 1000
                end_time = segment[1] * 1000
                
                # mute the segment
                audio = audio.overlay(AudioSegment.silent(duration=end_time-start_time), position=start_time)
    
    # save the muted audio to disk
    muted_audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "muted_audio.wav")
    audio.export(muted_audio_path, format="wav")
    
    # merge the muted audio with the original video
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.mp4")
    command = f"ffmpeg -i {video_path} -i {muted_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -strict -2 -y {output_path}"
    subprocess.call(command, shell=True)
    
    # start the live stream with the muted video
    return send_file(output_path, mimetype="video/mp4")

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)

# Set the upload folder path and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template("index.html", video_path=None)

@app.route('/', methods=['POST'])
def upload_video():
    # Check if a file was uploaded
    if 'video-file' not in request.files:
        return redirect(request.url)

    video_file = request.files['video-file']

    # Check if the file is empty
    if video_file.filename == '':
        return redirect(request.url)

    # Check if the file has an allowed extension
    if not allowed_file(video_file.filename):
        return redirect(request.url)

    # Save the file to the upload folder
    video_file.save(os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename))

    # Redirect to the video playback page
    return redirect(url_for('play_video', filename=video_file.filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/play/<filename>')
def play_video(filename):
    # Get the path of the uploaded video file
    video_path = url_for('uploaded_file', filename=filename)

    return render_template("index.html", video_path=video_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

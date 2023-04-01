import subprocess

# Replace with the path of your video file
video_path = '1.mp4'

# Replace with the RTMP URL of your streaming service
rtmp_url = 'rtmp://localhost/live/stream'

ffmpeg_cmd = ['ffmpeg', '-re', '-i', video_path, '-c:v', 'libx264', '-preset', 'veryfast', '-c:a', 'aac', '-f', 'flv', rtmp_url]

try:
    # Start the FFmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = ffmpeg_process.communicate()

    # Print FFmpeg output and error messages
    print(output.decode('utf-8'))
    print(error.decode('utf-8'))

except Exception as ex:
    print(ex)
    # If the user interrupts the script with Ctrl+C, stop the FFmpeg process
    ffmpeg_process.terminate()

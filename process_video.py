import cv2
import pytesseract
import multiprocessing
import logging
import pandas as pd
import ast

# Configure logging
# logging.basicConfig(filename='processing.log', level=logging.INFO,
#                     format='%(asctime)s [%(levelname)s]: %(message)s')

# PyTesseract configurations
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update the path to Tesseract executable
tessdata_dir = '/opt/tessdata'

# Constants for the number of worker processes and threads
NUM_CORES = multiprocessing.cpu_count()
NUM_PROCESSES = max(1, NUM_CORES - 2)  # You can adjust the number of processes as needed
NUM_THREADS = max(1, NUM_CORES)       # You can adjust the number of threads as needed


def extract_frames(video_path, interval, frame_queue):
    """
    Extract frames from a video and put them in the frame_queue.
    """
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % interval == 0:
            frame_queue.append([frame, frame_index])  # Put both frame and index into the queue
        frame_index += 1

    cap.release()
    return frame_queue


def preprocess(frame):
    """
    Preprocess an individual video frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray_frame, (5, 5), 0)


def process_frame(frame, frame_index, WORDS_TO_MUTE):
    """
    Process an individual frame: preprocess, detect text, and add to text_queue.
    """
    img = preprocess(frame)
    words = pytesseract.image_to_data(img, config=tessdata_dir, output_type=pytesseract.Output.DICT)
    masked_words_box = []
    masked_words = []

    for i in range(len(words['text'])):
        if words['text'][i].lower() in WORDS_TO_MUTE:
            #print(f"Masked word found in frame: {frame_index} with Pytesseract: {words['text'][i]}")
            x, y, w, h = words['left'][i], words['top'][i], words['width'][i], words['height'][i]
            masked_words_box.append((x, y, w, h))
            masked_words.append(words['text'][i].lower())

    #text_queue.put(masked_words)
    return masked_words_box, frame_index, masked_words

def draw_masks(frame, mask_df):
    """
    Draw masks on the frame based on the detected words.
    """
    for _, row in mask_df.iterrows():
        bounding_boxes = row['bounding_box']
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)  # Draw a filled black rectangle
    return frame



def save_processed_video(video_path, output_path, text_queue):
    """
    Process the video: extract frames, detect text, draw masks, and save the processed video.
    """
    cap = cv2.VideoCapture(video_path)

    # Get the original video's width, height, and frame rate
    width = int(cap.get(3))
    height = int(cap.get(4))
    frame_rate = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask_df = text_queue[text_queue['frame_index'] == count]
        frame_with_masks = draw_masks(frame.copy(), mask_df)  # Create a copy of the frame with masks drawn

        out.write(frame_with_masks)  # Write the frame to the output video
        count = count + 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video_path = 'test4.mp4'  # Replace with your input video file path
    output_video_path = 'test4_with_masks.mp4'  # Specify the output video file path
    interval = 1
    # Define words to mute
    WORDS_TO_MUTE = [
        "coward", "violent", "pray for you", "survive the war", "righteousness",
        "terrible wrongs", "bad men", "terrorizing", "stand", "paid a price for it",
        "help you", "talk", "ashamed", "love", "sins", "Red", "owns me", "Miss Jenna",
        "faith", "Channel"
    ]
    # Queues for frames and detected text
    text_queue = list()
    frame_queue = list()
    frame_queue = extract_frames(input_video_path, interval, frame_queue)
    for frame_info in frame_queue:
        frame = frame_info[0]
        frame_index = frame_info[1]
        masked_words_box, frame_index, masked_words = process_frame(frame, frame_index, WORDS_TO_MUTE)
        info = dict()
        info['frame_index'] = int(frame_index)
        info['bounding_box'] = masked_words_box if len(masked_words_box) else []
        info['mask_word'] = masked_words
        text_queue.append(info)
    text_queue = pd.DataFrame(text_queue)
    text_queue.to_csv("out4.csv", index=False)
    # Call the function with the sample data
    #text_queue = interpolate_frames_and_drop_duplicates(text_queue)
    save_processed_video(input_video_path, output_video_path, text_queue)


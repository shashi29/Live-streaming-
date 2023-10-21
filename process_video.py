import cv2
import pytesseract
import multiprocessing
import logging
import pandas as pd
import ast
import concurrent.futures
import os
import numpy as np

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
            print(f"Masked word found in frame: {frame_index} with Pytesseract: {words['text'][i]}")
            x, y, w, h = words['left'][i], words['top'][i], words['width'][i], words['height'][i]
            masked_words_box.append((x, y, w, h))
            masked_words.append(words['text'][i].lower())

    # Draw bounding boxes on the frame and save it
    # frame_with_boxes = frame.copy()
    # for (x, y, w, h) in masked_words_box:
    #     cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), -1)  # Green bounding box

    # save_path = os.path.join("save_frame", f"frame_{frame_index}.png")
    # cv2.imwrite(save_path, frame_with_boxes)

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

def fill_empty_rows(data):
    last_non_empty_mask_word = []
    last_non_empty_bounding_box = []
    last_rows = None

    # Iterate through the DataFrame to identify the last rows with non-empty 'mask_word'
    for index, row in data.iterrows():
        if len(row['mask_word']) > 0:
            last_rows = index

    for index, row in data.iterrows():
        mask_word = row['mask_word']
        bounding_box = row['bounding_box']

        if len(mask_word) > 0:
            last_non_empty_mask_word = mask_word
            last_non_empty_bounding_box = bounding_box

        if index < last_rows:
            data.at[index, 'mask_word'] = last_non_empty_mask_word
            data.at[index, 'bounding_box'] = last_non_empty_bounding_box

    return data

def process_frames_in_parallel(frame_queue, WORDS_TO_MUTE):
    num_worker_threads = os.cpu_count()//2

    # Define a function to process a single frame
    def process_frame_worker(frame_info):
        frame = frame_info[0]
        frame_index = frame_info[1]
        masked_words_box, frame_index, masked_words = process_frame(frame, frame_index, WORDS_TO_MUTE)
        info = dict()
        info['frame_index'] = int(frame_index)
        info['bounding_box'] = masked_words_box if len(masked_words_box) else []
        info['mask_word'] = masked_words
        return info

    # Create a ThreadPoolExecutor with the calculated number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Process frames in parallel
        futures = [executor.submit(process_frame_worker, frame_info) for frame_info in frame_queue]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

        # Retrieve the results from completed tasks
        text_queue = []
        for future in futures:
            result = future.result()
            text_queue.append(result)

    # Convert the results into a DataFrame
    text_queue_df = pd.DataFrame(text_queue)
    return text_queue_df


def find_range(num, ranges):
    for start, end in ranges:
        if num >= start and num <= end:
            return [start, end]
    return None, None

# Function to fill missing frames
def fill_missing_frames(frame_ranges, df):
    filled_df = df.copy()  # Create a copy of the DataFrame
    for index, rows in df.iterrows():
        bounding_box = rows['bounding_box']#ast.literal_eval(rows['bounding_box'])
        try:
            if len(bounding_box) == 0:
                start, end = find_range(index, frame_ranges)
                if start is not None and end is not None:
                    df_filtered = df.loc[start:end]
                    df_filtered = df_filtered[df_filtered['bounding_box'] != '[]']
                    unique_data = df_filtered.drop_duplicates(subset=['mask_word'])
                    replace_bounding_box = list()
                    replace_mask_word = list()
                    for bounding_box, mask_word in zip(unique_data['bounding_box'].tolist(), unique_data['mask_word'].tolist()):
                        #replace_bounding_box.extend(ast.literal_eval(bounding_box)) 
                        #replace_mask_word.extend(ast.literal_eval(mask_word))
                        replace_bounding_box.extend(bounding_box) 
                        replace_mask_word.extend(mask_word)
                    if replace_mask_word != []:
                        print(replace_bounding_box, replace_mask_word)
                        for i in range(index, end + 1):
                            filled_df.at[i, 'bounding_box'] =  replace_bounding_box
                            filled_df.at[i, 'mask_word'] =  replace_mask_word
        except Exception as ex:
            continue
    return filled_df


def find_similar_frame_ranges(video_path, threshold=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frame_ranges = []
    start_frame = None
    prev_frame = None

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop when no more frames are available

        # Convert the frame to grayscale for comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_frame.shape  # Get the height and width of the frame

        if prev_frame is not None:
            # Calculate the absolute difference between the current and previous frames

            # Ensure the frames have the same dimensions for comparison
            prev_frame = prev_frame[:height, :width]
            gray_frame = gray_frame[:height, :width]

            frame_diff = cv2.absdiff(prev_frame, gray_frame)

            # Calculate the mean value of the frame difference
            mean_diff = frame_diff.mean()

            if mean_diff <= threshold:
                if start_frame is None:
                    start_frame = frame_index - 1  # Start a new range
                end_frame = frame_index  # Update the end frame
            elif start_frame is not None:
                frame_ranges.append((start_frame, end_frame))  # End the current range
                start_frame = None

        prev_frame = gray_frame.copy()
        frame_index += 1

    cap.release()

    return frame_ranges

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
    save_processed_video(input_video_path, output_video_path, text_queue)

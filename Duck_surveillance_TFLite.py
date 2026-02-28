import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import subprocess
from time import sleep
from datetime import datetime
import RPi.GPIO as GPIO
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from typing import List

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN)


# Email parameters
subject = 'Security Alert: A motion has been detected'
bodyText = """\
Hi,
A dangerous species has been detected in your duck farm.
Please check your Gmail sent from raspberry pi security system.
Regards,
AS Tech-Workshop
"""
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
USERNAME = 'haiyangh61@gmail.com'
PASSWORD = 'fjuqthmjaieqhouq' # 16-digit App Password
RECEIVER_EMAIL = 'huanghaiyang720@gmail.com'


# Video filename and path
filename_part1 = "surveillance"
file_ext = ".mp4"
filepath = "/home/pi/tflite-custom-object-bookworm-main/"
tflite_path = "/home/pi/tflite-custom-object-bookworm-main/best.tflite"
threshold_prob = 0.8

def inference(image: np.ndarray,
			  model: Interpreter,
			  threshold: float = 0.8) -> (np.ndarray, List[str], List[float]):
	"""
	Yolo inference is performed on input image
	The result of the detection box is written back to image and returned.
	:param model: tensorflow lite model
	:param image: image to be detected
	:param threshold: minimum threshold of category
	:return: detection result includes image, class list and probability list
	"""


	# Image resize and normalization
	input_data = cv2.resize(image, (640, 640))
	input_data = input_data / 255.0
	input_data = np.expand_dims(input_data, axis=0).astype(np.float16)


	# Yolov5s model inference
	input_details = model.get_input_details()
	output_details = model.get_output_details()

	model.set_tensor(input_details[0]['index'], input_data)
	model.invoke()
	output_data = model.get_tensor(output_details[0]['index'])


	# Output data analyzation
	num_boxes = output_data.shape[1]
	boxes, scores, classes = [], [], []

	for i in range(num_boxes):
		if output_data[0, i, 4] > threshold: # Confidence threshold
			box = output_data[0, i, :4]
			x, y, w, h = box
			# Convert boundingbox coordinates back to the original image's scale
			x *= image.shape[1]
			y *= image.shape[0]
			w *= image.shape[1]
			h *= image.shape[0]
			# Convert the bbox coordinates from center coordinates to top-left and bottom-right corner format
			boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
			scores.append(output_data[0, i, 4])
			class_id = np.argmax(output_data[0, i, 5:])
			classes.append(class_id)


	# Initialization of return values
	valid_scores = []
	valid_class_names = []


	# Non-maximum supression is enabled only when a bounding box exists
	if len(boxes) > 0:
		indices = tf.image.non_max_suppression(boxes,
							scores,
							max_output_size=num_boxes,
							iou_threshold=0.5,
							score_threshold=threshold)
		# Adding boundingbox
		for i in indices:
			box = boxes[i]
			score = scores[i]
			class_id = classes[i]
			class_name = class_idx2name[class_id] # Class name


			cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
			cv2.putText(image, f'{class_name} {score:.2f}', (int(box[0]), int(box[1] - 10)), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


			valid_scores.append(float(score))
			valid_class_names.append(class_name)


	return image, valid_class_names, valid_scores


def preprocess_video(input_video_path: str, output_video_path: str, scale_percent: int = 75) -> (List[str], List[float]):
	"""
	preprocess video
	:param input_video_path: input video address
	:param output_video_path: processed video address ， return empty string if not detected
	:param scale_percent: video scaling
	:return: detection result includes image, class list and probability list
	"""

	# Open the input video
	cap = cv2.VideoCapture(input_video_path)
	if not cap.isOpened():
		print("Error: Could not open input video.")
		return

	# Define the histogram subtraction
	backSub = cv2.createBackgroundSubtractorMOG2()

	# Get video properties
	original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)

	# Calculate the target size of video
	target_width = int(original_width * scale_percent / 100)
	target_height = int(original_height * scale_percent / 100)

	# Define the codec and create VideoWriter object with the target size
	# Compress the video frame
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_video_path, fourcc, fps // 2, (target_width, target_height), True)
	
	# Process each frame
	total_names = [] # Result of all detection categroies in preprocessed video
	total_scores = [] # The result scores of all detection types in preprocessed video
	while True:
		ret, frame = cap.read()
		if not ret:
			break

	# Resize the input video frame
	resized_frame = cv2.resize(frame, (target_width, target_height),
							interpolation=cv2.INTER_AREA)
	
	# Apply histogram subtraction
	fgMask = backSub.apply(resized_frame)
	foreground = cv2.bitwise_and(resized_frame, resized_frame, mask=fgMask)

	# Apply histogram equalization on each channel separately if it's a color video
	if len(frame.shape) == 3:
		# Convert to YUV
		yuv_frame = cv2.cvtColor(foreground, cv2.COLOR_BGR2YUV)
		# Equalize the histogram of the Y channel
		yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
		# Convert back to BGR
		equalized_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
	else:
		# If it's a grayscale video, apply histogram equalization directly
		equalized_frame = cv2.equalizeHist(foreground)

	equalized_frame, names, scores = inference(equalized_frame, interpreter, threshold_prob)
	total_names.extend(names)
	total_scores.extend(scores)

	# Write the preprocessed frame
	out.write(equalized_frame)
	# Release everything when done
	cap.release()
	out.release()
	print(f"Processed video saved to {output_video_path}")
	return total_names, total_scores


def send_email(file_to_send: str):
	"""
	send email
	:param file_to_send: send file path
	"""
	message = MIMEMultipart()
	message["From"] = USERNAME
	message["To"] = RECEIVER_EMAIL
	message["Subject"] = subject

	message.attach(MIMEText(bodyText, 'plain'))
	with open(file_to_send, "rb") as attachment:
		mimeBase = MIMEBase('application', 'octet-stream')
		mimeBase.set_payload(attachment.read())

	encoders.encode_base64(mimeBase)
	mimeBase.add_header('Content-Disposition', 
						f"attachment;filename={os.path.basename(file_to_send)}")

	message.attach(mimeBase)
	text = message.as_string()

	session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
	session.starttls()
	session.login(USERNAME, PASSWORD)
	session.sendmail(USERNAME, RECEIVER_EMAIL, text)
	session.quit()
	print("Email sent")


def capture_video():
	now = datetime.now()
	current_datetime = now.strftime("%d-%m-%Y_%H:%M:%S")
	filename = filename_part1 + "_" + current_datetime + file_ext
	video_path = filepath + 'newvideo.h264'
	converted_video_path = filepath + filename


	# Use libcamera-vid to record the video
	process = subprocess.Popen([
		'libcamera-vid',
		'-t', '0', # Indefinite recording
		'-o', video_path
	])


	# Start a loop that checks for motion and stops recording after 5 seconds of no motion
	print("Started recording")
	try:
		while True:
			if GPIO.input(11) == GPIO.LOW: # If no motion detected
				print("Waiting for no motion")
				sleep(5) # Wait for 5 seconds to confirm no motion
			if GPIO.input(11) == GPIO.LOW: # If still no motion detected
				print("No motion detected for 5 seconds, stopping recording")
			process.terminate() # Send SIGTERM to stop libcamera-vid
			process.wait() # Wait for the process to terminate
			break
		sleep(1) # Check for motion every second
	except subprocess.SubprocessError as e:
		print(f"Error stopping recording: {e}")


	# Convert h264 to mp4
	result = subprocess.run(['MP4Box','-add', video_path,converted_video_path], stderr=subprocess.PIPE)
	if result.returncode != 0:
		print("Failed to convert video to mp4")
		print(result.stderr.decode())
		return None

	processed_video_path = filepath + "processed_" + current_datetime + file_ext
	scale_percent = 75 # Scale down to 75% of the original size
	detected_names, _ = preprocess_video(converted_video_path, processed_video_path, scale_percent)


	# Clean up the original h264 and mp4 files
	remove_file(video_path)
	remove_file(converted_video_path)


	# An empty string is returned if the result is an empty list
	return processed_video_path if len(detected_names) > 0 else "", detected_names


def remove_file(path):
	if os.path.exists(path):
		os.remove(path)
		print(f"Removed file {path}")
	else:
		print(f"File {path} does not exist")


# Main code for method call
if __name__ == "__main__":
	# Load tflite model
	interpreter = tf.lite.Interpreter(model_path=tflite_path)
	interpreter.allocate_tensors()

	# Class_name to index
	class_name2idx = {
	"duck": 0,
	"crow": 1,
	"hawk and eagle": 2,
	"owl": 3,
	"fox": 4,
	"rat": 5,
	"weasel and stoat": 6,
	"badger": 7
	}

	# Index to class_name
	class_idx2name = {item[1]: item[0] for item in class_name2idx.items()}

	# Class_name to bbox color
	class_name2color = {}
	for idx in range(len(class_name2idx)):
		r = random.randint(0, 255) # Random value of the red channel
		g = random.randint(0, 255) # Random value of the green channel
		b = random.randint(0, 255) # Random value of the blue channel
		class_name2color[idx] = (r, g, b)
		
	# Method call
	while True:
		if GPIO.input(11) == GPIO.HIGH: # If motion detected
			print("Motion Detected")
			video_file, detected_names = capture_video()
			detected_ids = [class_name2idx[i] for i in detected_names]
			if video_file: # If video captured and processed successfully
				send_email(video_file)
				remove_file(video_file) # Remove the processed mp4 file
			sleep(1) # Sleep to prevent rapid re-checking.

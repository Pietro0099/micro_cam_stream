import serial
import numpy as np
import cv2
import threading
import struct
import time
import os
import glob
import yaml
import sys
import colorama
from colorama import Fore, Style


class ImageDisplay:
	def __init__(self, img_shape, labels, percent_out, labels_offset, latency_offset, box_shape, zoom_factor=1, rectangle_thickness=-1, text_color=(0, 0, 0), text_thickness=2, font_size=0.8, display_imgs="both", std_box_color=(255, 0, 0), pred_box_color=(0, 255, 0), latency_box_color=(0, 255, 0), color_type="uniform", use_unknown_class=False, confidence=None, display_latency=False):
		self.img_shape = img_shape
		self.image_data = None
		self.labels_data = None
		self.class_latency = None
		self.dsp_latency = None
		self.display_thread = None
		self.lock = threading.Lock()
		self.zoom_factor = zoom_factor
		self.labels = labels
		self.percent_out = percent_out
		self.labels_offset = labels_offset
		self.latency_offset = latency_offset
		self.rectangle_thickness = rectangle_thickness
		self.text_color = text_color
		self.text_thickness = text_thickness
		self.font_size = font_size
		self.display_imgs = display_imgs
		self.std_box_color = std_box_color
		self.pred_box_color = pred_box_color
		self.latency_box_color = latency_box_color
		self.color_type = color_type
		self.box_shape = box_shape
		self.display_latency = display_latency
		self.use_unknown_class = use_unknown_class
		self.confidence = confidence
		self.animation_duration = 0.5
		self.slide_completed = [False] * len(labels)
		self.prev_boxes = [None] * len(labels)

	def start(self):
		self.display_thread = threading.Thread(target=self.display_loop)
		self.display_thread.start()

	def stop(self):
		self.display_thread.join()

	def update_image(self, image_data, labels_data, class_latency=None, dsp_latency=None):
		with self.lock:
			self.image_data = image_data.copy()
			self.labels_data = labels_data
			self.class_latency = class_latency
			self.dsp_latency = dsp_latency
			self.slide_completed = [False] * len(labels_data)
			self.prev_boxes = [None] * len(labels_data)

	def display_loop(self):
		while True:
			with self.lock:
				if self.image_data is None:
					continue

				# Reshape image
				image = self.image_data.reshape(self.img_shape)

				# Create zoomed version of the image
				new_width = int(image.shape[1] * self.zoom_factor)
				new_height = int(image.shape[0] * self.zoom_factor)
				zoomed_image = cv2.resize(image, (new_width, new_height))

				# Get the predicted label
				pred = np.argmax(self.labels_data)

				if self.use_unknown_class and (self.labels_data[pred] < self.confidence):

					u_size, _ = cv2.getTextSize("Unkown", cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_thickness)
					cv2.rectangle(zoomed_image, (self.labels_offset[0], self.labels_offset[1] - u_size[1]), (self.labels_offset[0] + u_size[0], self.labels_offset[1]), self.std_box_color, self.rectangle_thickness, cv2.LINE_AA)
					cv2.putText(zoomed_image, "Unkown", (self.labels_offset[0], self.labels_offset[1]), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_color, self.text_thickness, cv2.LINE_AA)

				else:

					# Add labels as little rectangles in the top left corner
					for i, label in enumerate(self.labels):
						# Select uniform color or differentiate the max value box label color
						if self.color_type == "uniform":
							box_color = self.std_box_color

						elif self.color_type == "max":
							if i == pred:
								box_color = self.pred_box_color
							else:
								box_color = self.std_box_color
						# Display outputs as raw floats or truncated percentages
						if self.percent_out:
							label = f"{label}: {round((self.labels_data[i])*100, 2)}%"
						else:
							label = f"{label}: {self.labels_data[i]}"

						# Apply box labels offset and save previous box shape
						x = self.labels_offset[0]
						y = self.labels_offset[1] + i * self.box_shape[2]

						start_w = self.prev_boxes[i][2] if self.prev_boxes[i] is not None else 0
						end_w = int(self.box_shape[0]*self.labels_data[i])
						h = self.box_shape[1]

						# Start animation
						if not self.slide_completed[i] and start_w != end_w:
							# Calculate the transition progress based on the current time
							progress = min(time.time() % self.animation_duration, self.animation_duration) / self.animation_duration

							# Calculate the current width based on the progress
							current_w = int((1 - progress) * start_w + progress * end_w)
						else:
							current_w = end_w
							self.slide_completed[i] = True

						# Draw the sliding rectangle
						cv2.rectangle(zoomed_image, (x, y - h), (x + current_w, y), box_color, self.rectangle_thickness, cv2.LINE_AA)
						cv2.putText(zoomed_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_color, self.text_thickness, cv2.LINE_AA)

						# Update the previous box dimensions
						self.prev_boxes[i] = (x, y - h, current_w, y)

						# Check if all slide animations have completed
						if all(self.slide_completed):
							self.slide_completed = [False] * len(self.labels)

				# Display latency values
				if self.display_latency:

					class_latency_str = f"Classification Latency: {self.class_latency} ms"
					dsp_latency_str = f"DSP Latency: {self.dsp_latency} ms"

					v_spacing_lat = 20

					cl_lat_size, _ = cv2.getTextSize(class_latency_str, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_thickness)
					dsp_lat_size, _ = cv2.getTextSize(dsp_latency_str, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_thickness)

					x_lat = self.img_shape[1]*self.zoom_factor - cl_lat_size[0] - self.latency_offset[0]
					y_lat = self.img_shape[0]*self.zoom_factor - (cl_lat_size[1] + v_spacing_lat + dsp_lat_size[1]) - self.latency_offset[1]

					cv2.rectangle(zoomed_image, (x_lat, y_lat), (x_lat + cl_lat_size[0], y_lat + cl_lat_size[1]), self.latency_box_color, self.rectangle_thickness, cv2.LINE_AA)
					cv2.putText(zoomed_image, class_latency_str, (x_lat, y_lat + cl_lat_size[1]), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_color, self.text_thickness, cv2.LINE_AA)

					cv2.rectangle(zoomed_image, (x_lat, y_lat + cl_lat_size[1] + v_spacing_lat), (x_lat + dsp_lat_size[0], y_lat + cl_lat_size[1] + v_spacing_lat + dsp_lat_size[1]), self.latency_box_color, self.rectangle_thickness, cv2.LINE_AA)
					cv2.putText(zoomed_image, dsp_latency_str, (x_lat, y_lat + cl_lat_size[1] + v_spacing_lat + dsp_lat_size[1]), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_color, self.text_thickness, cv2.LINE_AA)


				# Convert cv2 image from BGR to RGB
				rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				rgb_zoomed_image = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB)

				# Choose which images to display
				if self.display_imgs == "both":
					cv2.imshow("Raw Image", rgb_image)
					cv2.imshow("Zoomed Image", rgb_zoomed_image)
				elif self.display_imgs == "raw":
					cv2.imshow("Raw Image", rgb_image)
				elif self.display_imgs == "zoom":
					cv2.imshow("Zoomed Image", rgb_zoomed_image)

				cv2.waitKey(1)


def receive_images(ser, img_shape, num_labels, display, receive_lat_data=False):
	while True:
		# Wait for bytes to become available
		while ser.in_waiting == 0:
			pass
		
		# Calculate the total number of bytes expected for the image
		total_bytes = img_shape[0] * img_shape[1] * img_shape[2]
		bytes_received = 0

		# Initialize an empty array for image data
		image_data = np.empty(total_bytes, dtype=np.uint8)

		# Wait until all bytes are received
		while bytes_received < total_bytes:
			# Check if data is available
			if ser.in_waiting > 0:
				# Read available data
				available_bytes = min(total_bytes - bytes_received, ser.in_waiting)
				data = ser.read(available_bytes)
				
				# Store the received data in the image_data array
				image_data.flat[bytes_received : bytes_received + available_bytes] = np.frombuffer(data, dtype=np.uint8)
				
				# Update the count of received bytes
				bytes_received += available_bytes


		# Read label scores
		label_scores = []
		if receive_lat_data:
			while ser.in_waiting < (4 * num_labels + 8):
				pass
		else:
			while ser.in_waiting < (4 * num_labels):
				pass
		for _ in range(num_labels):
			score_bytes = ser.read(4) # Assuming 4-byte float values
			if len(score_bytes) != 4:
				break
			score = struct.unpack('f', score_bytes)[0]
			label_scores.append(score)
		else:
			if len(label_scores) == num_labels and len(image_data) == img_shape[0] * img_shape[1] * img_shape[2]:
				if receive_lat_data:
					# Receive latency data (4 bytes, 2 values)
					classification_lat_bytes = ser.read(4)
					dsp_lat_bytes = ser.read(4)

					if len(classification_lat_bytes) == 4 and len(dsp_lat_bytes) == 4:
						classification_lat = struct.unpack('I', classification_lat_bytes)[0]
						dsp_lat = struct.unpack('I', dsp_lat_bytes)[0]

						display.update_image(image_data, label_scores, classification_lat, dsp_lat)
					else:
						continue
				else:
					display.update_image(image_data, label_scores)
			continue

		# Skip updating the display and continue to the next iteration
		continue


def load_configurations(file_path):
	# Ensure only one stream_config file exists
	config_file_paths = glob.glob(file_path)
	if len(config_file_paths) == 1:
		config_file_extension = os.path.splitext(config_file_paths[0])[1].lower()
		config_file_path = config_file_paths[0]
	else:
		raise Exception(Fore.RED + Style.BRIGHT + f"Multiple or no stream_config files found. Config files found: {config_file_paths}" + Style.RESET_ALL)

	# Ensure file is .yaml
	if config_file_extension in (".yaml", ".yml"):
		with open(config_file_path) as f:
			data = yaml.safe_load(f)
	else:
		raise ValueError(Fore.RED + Style.BRIGHT + f"Unsupported config file format. Please use YAML format. Found format: {config_file_extension}" + Style.RESET_ALL)

	if not data:
		raise ValueError(Fore.RED + Style.BRIGHT + "No configurations found in the YAML file." + Style.RESET_ALL)

	configurations = data.get("configurations")
	if not configurations or not isinstance(configurations, list):
		raise ValueError(Fore.RED + Style.BRIGHT + "Invalid configurations format. Please ensure configurations are set up correctly." + Style.RESET_ALL)

	if len(configurations) == 1:
		return configurations[0]
	else:
		print("Multiple configurations found. Please choose one:")
		for i, config in enumerate(configurations, start=1):
			print(f"{i}. {config.get('name', f'Configuration {i}')}")
		while True:
			choice = input(Fore.WHITE + Style.BRIGHT + "Enter the number of the desired configuration: " + Style.RESET_ALL)
			try:
				choice = int(choice)
				if 1 <= choice <= len(configurations):
					return configurations[choice - 1]
				else:
					print(Fore.YELLOW + Style.BRIGHT + "Invalid choice. Please try again." + Style.RESET_ALL)
			except ValueError:
				print(Fore.YELLOW + Style.BRIGHT + "Invalid input. Please enter a number." + Style.RESET_ALL)




if __name__ == "__main__":

	# Read the config file
	colorama.init()
	print(Fore.MAGENTA + Style.BRIGHT + "Welcome to Serial CAM live stream!" + Style.RESET_ALL)
	print("Setting up configuration...")


	# Load configuration
	try:
		config = load_configurations("stream_settings.*")

		image_settings = config["image_settings"]
		serial_port_settings = config["serial_port_settings"]
		visualization_settings = config["visualization_settings"]
		prediction_settings = config["prediction_settings"]
		
		serial_port = serial_port_settings["serial_port"]
		baud_rate = serial_port_settings["baud_rate"]

		img_shape = image_settings["img_shape"]
		labels = image_settings["labels"]

		print("Configuration loaded successfully.")


	except Exception as e:
		print(Fore.YELLOW + Style.BRIGHT + f"Error: {e}" + Style.RESET_ALL)
		if sys.stdin.isatty():
			input(Fore.YELLOW + Style.BRIGHT + "Press Enter to exit..." + Style.RESET_ALL)


	# Sart serial
	print(f"Connecting to serial port: {serial_port}...")

	try:
		ser = serial.Serial(serial_port, baud_rate, timeout=1)
		print("Stream will begin as soon as data from the serial port becomes available.")

	except Exception as e:
		print(Fore.RED + Style.BRIGHT + f"Error: Failed to connect to serial port: {serial_port}" + Style.RESET_ALL)
		if sys.stdin.isatty():
			input(Fore.YELLOW + Style.BRIGHT + "Press Enter to exit..." + Style.RESET_ALL)


	# Start Streaming if serial connection was successfull
	if "ser" in globals():

		try:
			# Declare object for image streaming
			display = ImageDisplay(
				img_shape=img_shape,
				labels=labels,
				labels_offset=visualization_settings["labels_offset"],
				latency_offset=visualization_settings["latency_offset"],
				zoom_factor=visualization_settings["zoom_factor"],
				rectangle_thickness=visualization_settings["rectangle_thickness"],
				text_color=visualization_settings["text_color"],
				text_thickness=visualization_settings["text_thickness"],
				font_size=visualization_settings["font_size"],
				display_imgs=visualization_settings["display_imgs"],
				color_type=visualization_settings["color_type"],
				std_box_color=visualization_settings["std_box_color"],
				pred_box_color=visualization_settings["pred_box_color"],
				latency_box_color=visualization_settings["latency_box_color"],
				box_shape=visualization_settings["box_shape"],
				display_latency=prediction_settings["display_latency"],
				percent_out=prediction_settings["percent_out"],
				use_unknown_class=prediction_settings["use_unknown_class"],
				confidence=prediction_settings["confidence"],
			)

			# Begin thread
			display.start()

			receive_thread = threading.Thread(target=receive_images, args=(ser, img_shape, len(labels), display, prediction_settings["latency_from_ser"]))
			receive_thread.start()

			receive_thread.join()
			display.stop()

			ser.close()
			cv2.destroyAllWindows()

		except Exception as e:
					print(Fore.YELLOW + Style.BRIGHT + f"Error: {e}" + Style.RESET_ALL)
					if sys.stdin.isatty():
						input(Fore.YELLOW + Style.BRIGHT + "Press Enter to exit..." + Style.RESET_ALL)
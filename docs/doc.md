# Technical details

The data sent from the microcontroller to the serial must follow these rules:
	image data (R-> G-> B) as unsigned 8 bit integer.
	classification data (class_1_score, class_2_score, ...) as float 32 bit.

  **OPTIONAL**	
		latency data (classification latency, dsp latency) as integer 32 bit.
		Note: You can select whether to display or not the latency data. In any case, if your microcontroller is sending lantecy data, the variable 'latency_from_ser' MUST be set to true.

Notes:
	- Data must NOT contain any other overhead(like info log messages from the microcontroller). The script will try to reject any spurious data but it might cause crash.
	- Image data is NOT REQUIRED to be sent all together, chunks of data are allowed as long as the total number of bytes matches the expected number of bytes to represent the image.
	- Classification data is REQUIRED to be sent as chunks of 4 bytes per class score.
	- Classification results and boxes are ONLY displayed on the zoomed image, since the raw image may be too small to contain the label boxes.



# Documentation for Variables


## 'image_settings':

- `img_shape`: List [height,width,channels]. Specifies the shape of the input images to be processed. It is expected to be a list of three integers representing the height, width, and number of channels of the image, respectively. Pls note that the script only works with RGB images. Do NOT use channels != 3

- `labels`: List [class_1,class_2,...]. Defines the categories or labels that the model can recognize. It is expected to be a list of strings, where each string represents a distinct label.



## 'serial_port_settings':

- `serial_port`: String. Specifies the serial port to be used for communication. It is expected to be a string indicating the name or identifier of the serial port, such as "COM3".

- `baud_rate`: Integer. Sets the baud rate for serial communication. It is expected to be an integer representing the number of bits per second.



## 'visualization_settings':

- `labels_offset`: List [x,y]. Specifies the offset or margin for displaying labels boxes on the output image. It is expected to be a list of two integers representing the horizontal and vertical offsets, respectively. The offset is calculated from the left-high corner.

- `latency_offset`: List [x,y]. Specifies the offset or margin for displaying latency boxes on the output image. It is expected to be a list of two integers representing the horizontal and vertical offsets, respectively. The offset is calculated from the right-low corner.

- `rectangle_thickness`: Integer. Sets the thickness of the rectangles used for drawing boxes. It is expected to be an integer value. Negative values may indicate filled rectangles.

- `text_color`: List [R,G,B]. Defines the color of the text displayed within the boxes. It is expected to be a list of three integers representing the RGB values of the color.

- `text_thickness`: Integer. Sets the thickness of the text displayed with the boxes. It is expected to be an integer value.

- `font_size`: Float. Specifies the size of the font used for displaying text. It is expected to be a floating-point number.

- `display_imgs`: String. Determines which images should be displayed. It is expected to be a string indicating the desired display mode, "both"(raw and zoomed image), "raw"(only raw image), or "zoom"(only zoom). 

- `zoom_factor`: Integer. Sets the zoom factor for displaying the zoomed image. It is expected to be an integer value representing the magnification level.

- `color_type`: String. Defines the color style for the label boxes. It is expected to be a string indicating the color style, "max"(use a different color for the label with the maximum score), "uniform"(use uniform colors for all the label boxes).

- `std_box_color`: List [R,G,B]. Specifies the color for drawing label boxes. It is expected to be a list of three integers representing the RGB values of the color.

- `pred_box_color`: List [R,G,B] or null. Specifies the color for drawing label box with maximum score. It is expected to be a list of three integers representing the RGB values of the color. Ignored if 'color_type' is "unifrom"(can be set to null or dummy values).

- `latency_box_color`: List [R,G,B] or null. Specifies the color for drawing latency box. It is expected to be a list of three integers representing the RGB values of the color. Ignored if 'color_type' is "latency_from_ser" or "display_latency" is false(can be set to null or dummy values).

- `box_shape`: List [max_width,height,vertical_spacing]. Defines the shape of the label boxes and the vertical spacing between boxes. It is expected to be a list of three integers representing the max_width, height, and vertical_spacing, respectively. The max_width represent the width for a label box with score = 1 (100% confidence). Width of the box will be shorter if score < 1 (less than 100% confidence). With of the box is max_width * class_score(raw float value).



## 'prediction_settings':

- `latency_from_ser`: Bool. Determines whether the system should expect the latency data('true') from the serial port. It is expected to be a boolean value (`true` or `false`).

- `display_latency`: Bool. Determines whether the system should display the latency data('true'). It is expected to be a boolean value (`true` or `false`). MUST be set to 'false' if 'latency_from_ser' is 'false'.

- `percent_out`: Bool. Determines whether the system should output the prediction results as percentages('true') or raw float values('false'). It is expected to be a boolean value (`true` or `false`).

- `use_unknown_class`: Bool. A boolean value that determines whether a post-processing method should be used to select an "Unknown" class when the maximum label score is below a certain threshold (confidence).

- `confidence`: Float or null. This attribute specifies the confidence threshold to be used when use_unknown_class is set to true. It sets the minimum confidence level required for the model to select a specific class as the prediction. It's important to note that for the model to resort to the "Unknown" class, the confidence must be strictly less than the maximum label score. This attribute is ignored if use_unknown_class is set to false.


## 'configurations':

- `name`: String. Name used for the specific configuration.



**Example stream_settings.yaml**

defaults: &defaults

  image_settings: &image_defaults
    img_shape: [96, 96, 3]
    labels: [class_1, class_2, class_3, class_4]

  serial_port_settings: &serial_port_defaults
    serial_port: COM3
    baud_rate: 115200

  visualization_settings: &visualization_defaults
    labels_offset: [10, 30]
    latency_offset: [10, 30]
    rectangle_thickness: -1
    text_color: [0, 0, 0]
    text_thickness: 2
    font_size: 0.8
    display_imgs: both
    zoom_factor: 10
    color_type: max
    std_box_color: [255, 0, 0]
    pred_box_color: [0, 255, 0]
    latency_box_color: [255, 255, 255]
    box_shape: [500, 20, 30]

  prediction_settings: &prediction_defaults
    latency_from_ser: true
    display_latency: true
    percent_out: true
    use_unknown_class: false
    confidence: null


configurations:

  - name: config1
    <<: *defaults

  - name: config2
    <<: *defaults
    visualization_settings:
      <<: *visualization_defaults
      color_type: uniform
      std_box_color: [0, 255, 127.5]
    prediction_settings:
      <<: *prediction_defaults
      percent_out: false
      display_latency: false

  - name: config3
    <<: *defaults
    visualization_settings:
      <<: *visualization_defaults
      std_box_color: [0, 0, 255]
      pred_box_color: [255, 127.5, 0]
    prediction_settings:
      <<: *prediction_defaults
      use_unknown_class: true
      confidence: 0.6
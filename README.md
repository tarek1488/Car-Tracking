# Car Tracking with Plate Number Detector

## Overview

This project focuses on detecting and tracking cars along with their plate numbers in real-time video streams. It combines state-of-the-art object detection and tracking techniques to ensure accurate and efficient performance.

## Features

- **Car Detection**: Utilizes YOLOv8 for accurate car detection in various environments.
- **License Plate Detection**: Integrates a custom-trained model for detecting and reading license plates.
- **Tracking**: Employs DeepSORT for robust multi-object tracking to maintain identity consistency across frames.
- **Real-Time Processing**: Optimized for real-time performance on video streams.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/tarek1488/Car-Tracking.git
    cd Car-Tracking
    ```

2. **Set Up the Environment**:
    - Create a virtual environment:
        ```bash
        python -m venv env
        ```
    - Activate the virtual environment:
        - On Windows:
            ```bash
            .\env\Scripts\activate
            ```
        - On macOS/Linux:
            ```bash
            source env/bin/activate
            ```
    - Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

3. **Download the Models**:
    - Download the pretrained YOLOv8 model and the custom plate detector model from the provided links and place them in the `models/` directory.

4. **Run the Application**:
    ```bash
    python main.py
    ```

## Usage

1. **Input Video**:
   - You can provide a video file by changing the input file path in main.py.
   - Run after that adding_missing_data.py.

2. **Output**:
   - Run Visualize.py to produce the output video.
   - The output will be saved in your directory as out.mp4.

## Demo

You can watch a demonstration of the project in action below:

[![Car Tracking with Plate Number Detector](Capture.png)](https://youtu.be/FRM_TJaJS7M)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

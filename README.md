# Real-Time Adjustable Face Mesh Detection with OpenCV and MediaPipe

This project demonstrates real-time face mesh detection using OpenCV and MediaPipe, with the added functionality of adjusting the face mesh drawing parameters (thickness and circle radius) in real-time. The script captures live video from your webcam, detects facial landmarks, and draws a face mesh on detected faces. The frame rate (FPS), drawing thickness, and circle radius are displayed on the screen.

## Features

- **Face Mesh Detection**: Detects facial landmarks and draws a mesh over the detected faces.
- **Real-Time Adjustment**: Adjust the thickness and circle radius of the face mesh lines and points using keyboard controls.
- **Parameters Display**: Displays the current Thickness, and circle radius values on the screen. 
- **Responsive Controls**: Control face mesh appearance in real-time:
  - Increase thickness: `W`
  - Decrease thickness: `S`
  - Increase circle radius: `A`
  - Decrease circle radius: `D`
  - Exit: `ESC`

## Installation

To run this project, you'll need the following dependencies:

### Requirements

- **Python** 3.x
- **OpenCV** (`cv2`)
- **MediaPipe**

You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Clone the repository

You can clone this repository using the following command:

```bash
git clone git@github.com:0xCurtis/face-recognition-py.git
```

## Usage

Once the dependencies are installed, you can run the script using your Python environment:

```bash
python main.py
```

### Webcam Setup

The script uses your default webcam. If you have multiple cameras, you can adjust the `cv2.VideoCapture()` parameter to switch between them.

### Key Features in the Script

- **Face Mesh Detection**: The script uses MediaPipe's FaceMesh solution to detect up to two faces and draw face mesh contours with customizable thickness and circle radius.
- **Adjustable Parameters**: Use the `W`, `S`, `A`, and `D` keys to change the drawing thickness and circle radius in real-time:
  - `W`: Increase line thickness
  - `S`: Decrease line thickness
  - `A`: Increase circle radius
  - `D`: Decrease circle radius
  - `ESC`: Exit the program
- **FPS Calculation**: The frame rate is calculated and displayed on the video stream.
- **Exit Condition**: Press the "ESC" key to close the program.

### Customizable Parameters

- `max_num_faces`: The maximum number of faces to detect and process at a time (default is 2).
- `frameWidth` and `frameHeight`: Set the resolution of the webcam feed (default is 1920x1080).
- `cap.set(10, 150)`: Adjusts the brightness of the video feed (default is 150).
- `drawSpec.thickness`: Initial thickness of the lines drawn for the face mesh (default is 1, adjustable with `W` and `S` keys).
- `drawSpec.circle_radius`: Initial radius of the circles drawn at each face mesh landmark (default is 1, adjustable with `A` and `D` keys).

## Example Output

The script displays the webcam feed with the face mesh drawn over detected faces. The FPS, current thickness, and circle radius are shown in the top left corner. As you press `W`, `S`, `A`, or `D`, you can observe real-time adjustments to the face mesh.

## Future Improvements

- Add the ability to save the video with adjustable face meshes.
- Optimize for detecting more than two faces simultaneously with recognizing and drawing multiple face meshes.
- Implement additional face mesh drawing styles for more customization.

## References

- [MediaPipe Documentation](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to fork the project and make your own improvements. Contributions are welcome!

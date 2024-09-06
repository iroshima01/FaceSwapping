FACE SWAPPING APPLICATION

This Python application automatically swaps faces between pairs of images using OpenCV and dlib. It implements sophisticated image processing techniques to detect faces, extract facial landmarks, perform Delaunay triangulation, and seamlessly clone the swapped faces to create natural-looking results.

------Features------

-Automatic Face Detection: Uses dlib's frontal face detector to locate faces within any given image.

-Facial Landmark Detection: Employs dlib's shape_predictor to identify key facial points essential for accurate image manipulation.

-Delaunay Triangulation: Constructs triangles from the detected facial landmarks to map out the face structure, enabling precise transformations during the face swap.

-Seamless Cloning: Integrates swapped faces into the target images using OpenCV's seamlessClone function, ensuring smooth and visually appealing results.

------Prerequisites------
Before you can run this application, you need to have the following installed:

Package       Version
------------- ---------
dlib          19.24.4
numpy         1.21.6
opencv-python 4.10.0.84
pip           23.2.1
setuptools    68.0.0
wheel         0.41.2
python	      3.7		

You can install these packages using pip:
pip install numpy opencv-python dlib

Install the shape predictor model from the link below:
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

------Installation------
Clone the repository: Download the zip file.

Navigate to the project directory: Change directory to the project's root folder in your terminal or command prompt.

Install dependencies: Install the versions which are mentioned in Installation part.

------Project Structure------

main.py: The main Python script that you run to perform face swapping.
requirements.txt: Lists all the necessary Python packages.
README.md: Provides project documentation.
Usage
Follow these steps to run the application:

Prepare your images: Place all target images (in .jpg or .png format) into a folder, the location doesn't matter (Additionally, i have provided an image folder for trying).

Run the script: Execute the main script from your terminal:

python main.py

View the results: Check the folder which you write in code to save the output images to see the swapped face images. Each output file is named using the format a_<image1>_and_<image2>.jpg, reflecting the source images used for the face swap.

------Output------

The swapped face images are saved in a directory you gave for saving the output images. Each image file is named according to the format a_<img1>_and_<img2>.jpg, where <img1> and <img2> are the names of the original images used for swapping.

# Segmentation
Course Computer Vision: Segmentation

**Academic Year**: 2013-2014 (2nd semester - 1st Master of Science in Engineering: Computer Science)

## About
A script counting the number of white blood cells in a given image using a Hough transform to detect circles (using Canny edge detection to detect the edges of the cells). For each found circle a feature vector (i.e. the average HSV color in the circle) is calculated. Finally, features are removed if they are located outside a manually determined interval.

## Use
<p align="center">
<img src="res/normal.jpg" width="430">
</p>
<p align="center">Input image</p>
<p align="center">
<img src="res/canny.png" width="430">
<img src="res/hough.png" width="430">
</p>
<p align="center">Canny edge detection - Hough transform </p>
<p align="center">
<img src="res/info.png" width="430">
<img src="res/result.png" width="430">
</p>
<p align="center">Feature vectors - Result</p>

# Scanning Whiteboard Contents
## Final Project of SCC0251-Digital Image Processing course
This is a project developed as the final project of the Digital Image Processing course(SCC0251) at the University of São Paulo (USP).
### Students
* [David Souza Rodrigues](https://github.com/bicanco);
* [Marcos Wendell Souza de Oliveira Santos](https://github.com/MarcosWendell);
### Objective
* The goal of this application is the digitalization of whiteboard contents capture. This application locates the boundary of the whiteboard, rectifies the geometry distortion and corrects the non-uniform illumination.
### Application Functionalities
* The application will read a whiteboard image, remove its surroundings, centralize the whiteboard content and adjust image luminosity as to facilitate the comprehension of the image.
### Images Source
* The application will be developed based on the images available on [the pantheon project website](https://clouard.users.greyc.fr/Pantheon/experiments/whiteboard-scanning/index-en.html), plus images captured by the students using black, blue, red and green markers while using flash on the camera.
### Digital Images Processing Techniques
* Edges' detection: Using the differential filter with a convolution operation;
* Horizontal and Vertical lines' detection: A convolution with the proper filters may be used, as well as the Hough transform;
* Whiteboard's boundaries detection: Given the detected lines, identify the best possible match for the quadrilateral shape of the whitedoard's boundaries;
* Image's Geometry correction: Perform a 3D transformation to adjust the coordinates;
* Luminosity correction: Color enhancement to obtain a uniform background, removing light spots.

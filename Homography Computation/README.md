To run these modules, the following libraries are required:
1. Numpy
2. OpenCV
3. matplotlib
4. math

When running these modules, openCV will display images. In problem 1, each modified video frame will each show for 0.3 seconds minimum before the next one displays. Press 'esc' in these windows to end the operation early. In problem 2, the final stitched image will display after computation time. Press any key to end the operation from this point.

Both Project2_2.py and Project2_1.py have been independently tested to esnure there are no recursive dependencies and confirmed to work correctly. If any errors arise, please reach out to nnovak@umd.edu and I will address them accordingly, if they are issues with my code.

To run these files, ensure you have the installed dependencies and either run them from an IDE or in terminal, with pwd in the Project2 folder, type '$python3 Project2_###.py' where ### represents the number of the file you want to run.

Viewing outputs:
This folder contains a few outputs. It contains the warped image of 2,3,and 4 stitched together as well as the full stitch for problem 2. Further, it contains a modified video for problem 1. Both of these can be viewed to confirm the output. Further, there are commented out cv2.imshow() calls in problems 1 and 2 that will display other images such as the accumulator or the intermediate stitches. By default, Project2_1.py will write out a video of the modified frames, and Project2_2.py will write out an image of the final full stitch.


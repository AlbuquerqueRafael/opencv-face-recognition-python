#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os

# Path of the files that you want to resize
path = "training-data/Experimento1/caseB"
path_content = os.listdir(path)

for image_name in path_content:
    print (image_name)
    image_path  = path + '/' + image_name
    image = cv2.imread(image_path)

    resized_image = cv2.resize(image, (800, 600))

    #
    cv2.imwrite(image_path, resized_image)

cv2.destroyAllWindows()

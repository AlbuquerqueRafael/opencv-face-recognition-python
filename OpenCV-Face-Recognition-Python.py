#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as
#it is needed by OpenCV face recognizers
import numpy as np
import random

subjects = ["", "Anne", "Tom", "Idris"]
IMG_INDEX = 1
CONFIDENCE_INDEX = 0


def detect_face(img, image_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    cv2.imshow(image_name , cv2.resize(gray[y:y+h,x:x+w], (200, 200)))
    cv2.waitKey(100)
    # resize the faces
    resized_face = cv2.resize(gray[y:y+h,x:x+w], (200, 200))


    #return only the face part of the image
    return resized_face, faces[0]


#Prepare training data
def prepare_training_data(data_folder_path, arrayTest, numberOfTrainingImages):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []
    imagesUsed = {}
    imagesUsed["caseA"] = []
    imagesUsed["caseB"] = []
    imagesUsed["caseC"] = []

    for dir_name in dirs:
        label = 0

        if dir_name == "caseA":
            label = 1
        elif dir_name == "caseB":
            label = 2
        else:
            label = 3

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)
        print (subject_dir_path)

        #Exclude the faces that will be used as test of our training data
        subject_images_names = list(set(subject_images_names).difference(arrayTest))
        index = 0
        for image_name in subject_images_names:

            image_path  = subject_dir_path + '/' + image_name
            image = cv2.imread(image_path)

            #detect face
            face, rect = detect_face(image, image_name)

            if face is not None:
                index += 1
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)

                imagesUsed[dir_name].append(image_name)
            else:
                print ("Image not recognized" + image_name)

            if index == numberOfTrainingImages:
                break

    print ("Images used: ")
    print (imagesUsed)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)



def predict(test_img, image_name):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img, image_name)

    #predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]

    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)

    data = [confidence, img]
    return data, label_text

def defineTestImagesArray(numberOfTestImages):
    #Set the seed for the experiment
    random.seed(10)
    numberOfImages = 30
    arrayTest = []

    i = 0
    #Be careful here, this could generate a runtime
    while i != numberOfTestImages:
        value = random.randint(1, numberOfImages + 1)
        value = str(value) + '.jpg'
        if (value not in arrayTest):
            arrayTest.append(value)
            i += 1
    return arrayTest

def getPreparedData(numberOfTrainingImages, numberOfTestImages):

    arrayTest = defineTestImagesArray(numberOfTestImages)
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data/Experimento1", arrayTest, numberOfTrainingImages)
    print("Data prepared")

    return faces, labels, arrayTest

def predictTest(arrayTest, path, testingPerson):
    acertos = 0
    erros = 0

    for image_name in arrayTest:
        image_path  = path + '/' + image_name
        test_img = cv2.imread(image_path)
        predicted, label_text = predict(test_img, image_name)
        predicted_img = predicted[IMG_INDEX]
        confidence = predicted[CONFIDENCE_INDEX]
        print("Prediction complete")

        if (label_text == testingPerson):
            acertos += 1
        else:
            erros += 1

        cv2.imshow("output -> acurracy: " + str(confidence), cv2.resize(predicted_img, (400, 500)))

    return acertos, erros



def main():

    # Para o nosso exp, use 25 ou 5
    numberOfTrainingImages = 25

    numberOfTestImages = 5
    faces, labels, arrayTest = getPreparedData(numberOfTrainingImages, numberOfTestImages)
    face_recognizer.train(faces, np.array(labels))
    print("Data trained")
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    print("Predicting images...")
    path = "training-data/Experimento1/caseA"
    acertos, erros = predictTest(arrayTest, path, "Anne")

    print ("Result: ")
    print ("acertos: " + str(acertos))
    print ("erros: "  + str(erros))
    print ("total: " + str(acertos + erros))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# face_recognizer_eigen = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()

main()

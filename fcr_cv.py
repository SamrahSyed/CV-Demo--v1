# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:34:03 2021

@author: samrah.asif
"""

#import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time
from matplotlib import pyplot as plt
#import uploadfiles
#from uploadfiles import app

#UPLOAD_FOLDER = 'uploads/'
#UPLOAD_FOLDER = uploadfiles.upload_file()

#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)`
#app = Flask(__name__)
def fcr(path, UPLOAD_FOLDER):
    UPLOAD_FOLDER=
    imagePaths = list(paths.list_images((UPLOAD_FOLDER)))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print(imagePath)
        name = imagePath.split(os.path.sep)[-1]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    #save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #print(data)
    #use pickle to save data into a file for later use
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()

    #find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    
    # image = cv2.imread('test/aj4.jpeg')
    image = cv2.imread(path)
    #cv2.imshow("Frame", image)
    #print(image)
    
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for haarcascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    
    encodings = face_recognition.face_encodings(rgb)
    names = []
    
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
        encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
     
     
            # update the list of names
            names.append(name)
            #print(xyz)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # Window name in which image is displayed
                #window_name = 'image'
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                 0.75, (0, 255, 0), 2)
     #   plt.figure(image)
     #   plt.imshow(image)
        #plt.axis('off') # if you want to remove coordinates
        #data = random.random((5,5))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #image.set_cmap('hot')
        plt.axis('off') 
        cv2.imwrite(path, image)
        return #"target_image"
      #  plt.savefig("test.png", bbox_inches='tight')
      #  cv2.show()
#        cv2.imshow("Frame", image)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        cv2.waitKey(1)
        
#fcr('test/aj1.jpeg','uploads/')

#import matplotlib.image as mpimg #If any issues in output, uncomment thispip ipython
#import numpy as np
from PIL import Image
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr
import numpy as np
import os.path
#import matplotlib.pyplot as plt
import cv2


""" class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name = self.classify(cropped_image)['class_name']
        return self.draw(image, boxes2D) """
class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name = self.classify(cropped_image)['class_name']
        return self.draw(image, boxes2D)
        
detect = EmotionDetector()
def emd_test(pic):
    
            
    detect = EmotionDetector()
    # you can now apply it to an image (numpy array)

    #image = cv2.imread('static\cry2.jpeg')
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    path = "static/uploads"
    
    # Join various path components
    #image = np.array(Image.open(os.path.join(path,  pic)))
    image = np.array(Image.open(pic))
    np_img = np.array(image)
    frame = detect(np_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pic,frame)
    return

    #To display image
    #cv2.imshow('image', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)
   
   
   
    #target_image_emd=cv2.imwrite('static/uploads/demo_emd.jpeg', frame)
    #plt.savefig(frame)
    #return target_image_emd
    
#emd_test("bopity.jpeg")

    #image = Image.open(path,  "cry2.jpeg")
    #cv2.imread(image)

    

    #.save("emd_demo.jpeg")
    #cv2.imwrite("static/image.jpeg",image)
    #plt.savefig(image)
    
    #cv2.imshow('image', frame)
#target_image = cv2.imwrite('.\static\uploads\demo987.jpeg',frame)
  #  return target_image """
    #image.save("emd_demo.jpeg")
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #cv2.waitKey(1)
    #return target_image
import time

import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx

class VideoCamera(object):
    SHRINK_RATIO = 0.25
    FPS = 10
    FRAME_RATE = 5
    WIDTH = 640
    HEIGHT = 480
    
    def __init__(self):
       #capturing video
       self.cap = cv2.VideoCapture(0)
       self.cap.set(3,VideoCamera.WIDTH)
       self.cap.set(4,VideoCamera.HEIGHT)
       self.cap.set(5,VideoCamera.FPS)
       self.cap.set(7,VideoCamera.FRAME_RATE)
    
#    def __del__(self):
        #releasing camera
#        self.video.release()
 
    def get_frame(self):  
        # Load the model
        net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
        # Compile the model for faster speed
        net.hybridize()
        
        # Load the webcam handler
#        cap = cv2.VideoCapture(0)
#        time.sleep(1) ### letting the camera autofocus
        
        axes = None
        NUM_FRAMES = 200 # you can change this
        for i in range(NUM_FRAMES):
            # Load frame from the camera
            ret, frame = self.cap.read()
            rescale_frame = cv2.resize(frame, None, fx= VideoCamera.SHRINK_RATIO, fy=VideoCamera.SHRINK_RATIO)        
#            ret, frame = cap.read()
            # Image pre-processing
            frame = mx.nd.array(cv2.cvtColor(rescale_frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)
        
            # Run frame through network
            class_IDs, scores, bounding_boxes = net(rgb_nd)
        
            # Display the result
            img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
            gcv.utils.viz.cv_plot_image(img)
            cv2.waitKey(5)
            
            
            ret, jpg = cv2.imencode('.jpg', frame)
            return jpg.tobytes()
            
        #cap.release()
        #cv2.destroyAllWindows()
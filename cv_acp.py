import pip
pip.main(["install","matplotlib"])
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
from gluoncv.utils import try_import_cv2
from ipywidgets import Video,Layout
import re

from gluoncv.utils.filesystem import try_import_decord


def TSN_PIC(img):
    img = image.imread(img)
 
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(img.asnumpy())

    transform_fn = transforms.Compose([
    video.VideoCenterCrop(size=224),
    video.VideoToTensor(),
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_list = transform_fn([img.asnumpy()])
    net = get_model('vgg16_ucf101', nclass=101, pretrained=True)
    pred = net(nd.array(img_list[0]).expand_dims(axis=0))
    classes = net.classes
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input image is classified to be')
    results=[]
    for i in range(topK):
        datadict = {
             'header': ('\t%s'%(re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', ((classes[ind[i].asscalar()]).split('-',1)[0])))),
             'data': ('%.2f'%(nd.softmax(pred)[0][ind[i]].asscalar()*100)+'%')
        }  
        results.append(datadict)
    print (results)
    return results
#        result = ('\t%s - %.2f'%(classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100)+'%')
#        result.replace(" ", "_")
#        result.replace(" ", "")
#        datadict = {
#             'data': result
#        }  
#        results.append(datadict)
#    print (results)
#    return results
#    for i in range(topK):
#        results = []
#        result = ('\t%s - %.2f'%(classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100))
#
#        datadict = {
#             'data': result
#        }  
#        results.append(datadict.copy())
#    return results
   # return plt.show()
       #    print((result)) 
          #   'header': 
              #print (results)
    
#TSN_PIC('static/playing_celo.jpeg')
    

global final_pred
def TSN_VIDEO(vid):
    cv2 = try_import_cv2()
    cap = cv2.VideoCapture(vid)
    cnt = 0
    video_frames = []
    transform_fn = transforms.Compose([
    video.VideoCenterCrop(size=224),
    video.VideoToTensor(),
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = get_model('vgg16_ucf101', nclass=101, pretrained=True)
    while(cap.isOpened()):
        ret, frame = cap.read()
        cnt += 1
        if ret and cnt % 25 == 0:
            video_frames.append(frame)
        if not ret: break
    cap.release()
    if video_frames:
        video_frames_transformed = transform_fn(video_frames)
        final_pred = 0
        for _, frame_img in enumerate(video_frames_transformed):
            pred = net(nd.array(frame_img).expand_dims(axis=0))
            final_pred += pred
    final_pred /= len(video_frames)

    classes = net.classes
    topK = 5
    ind = nd.topk(final_pred, k=topK)[0].astype('int')
    print('The input video frame is classified to be')
    results=[]
    for i in range(topK):
        datadict = {
             'header': ('\t%s'%(re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', (classes[ind[i].asscalar()]))).split('-',1)[0]),
             'data': ('%.2f'%(nd.softmax(pred)[0][ind[i]].asscalar()*100)+'%')
        }  
        results.append(datadict)
    print (results)
    return results
#    results=[]
#    for i in range(topK):
#        results=('\t%s - %.2f'%(classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100))
#        datadict = {
#          #   'header': 
#             'data': results
#        }  
#        results.append(datadict)
##        results=results+" "
#    #print (results)
#    return results
   # return Video.from_file(vid,layout=Layout(width='80%', height='80%'))

  
#TSN_VIDEO('static/v_Basketball_g01_c01.mp4')
    


def I3D_VID(vid):
    decord = try_import_decord()
    vr = decord.VideoReader(vid)
    frame_id_list = range(0, 64, 2)
    video_data = vr.get_batch(frame_id_list).asnumpy()
    clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    model_name = 'i3d_inceptionv1_kinetics400'
    net = get_model(model_name, nclass=400, pretrained=True)
    pred = net(nd.array(clip_input))
    classes = net.classes
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input video frame is classified to be')
    results=[]
    for i in range(topK):
#       result = ('\t%s - %.2f'%(((classes[ind[i].asscalar()]).replace("_", " ")).capitalize(), (nd.softmax(pred)[0][ind[i]].asscalar()*100))+'%')
#        result.replace("_", " ")
        datadict = {
             'header': ('\t%s'%(((classes[ind[i].asscalar()]).replace("_", " ")).title()).split('-',1)[0]),
             'data': ('%.2f'%(nd.softmax(pred)[0][ind[i]].asscalar()*100)+'%')
        }  
        results.append(datadict)
    print (results)
    return results
#    results=[]
#    for i in range(topK):
#        results=('\t%s - %.2f'%(classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100))
#        datadict = {
#          #   'header': 
#             'data': results
#        }  
#        results.append(datadict)
#    #print (results)
#    return results
#        result=('\t%s - %.2f'%
#          (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100))
#        print(result) 
#    return Video.from_file(vid)

#I3D_VID('static/v_Basketball_g01_c01.mp4')



def SlowFast(vid):
    decord = try_import_decord()
    vr = decord.VideoReader(vid)
    fast_frame_id_list = range(0, 64, 2)
    slow_frame_id_list = range(0, 64, 16)
    frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)
    video_data = vr.get_batch(frame_id_list).asnumpy()
    clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (36, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    model_name = 'slowfast_4x16_resnet50_kinetics400'
    net = get_model(model_name, nclass=400, pretrained=True)
    pred = net(nd.array(clip_input))

    classes = net.classes
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input video frame is classified to be')
    results=[]
    for i in range(topK):
        datadict = {
             'header': ('\t%s'%(((classes[ind[i].asscalar()]).replace("_", " ")).title()).split('-',1)[0]),
             'data': ('%.2f'%(nd.softmax(pred)[0][ind[i]].asscalar()*100)+'%')
        }  
        results.append(datadict)
    print (results)
    return results
#    results=[]
#    for i in range(topK):
#        result=('\t%s - %.2f'%(classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100))
#        datadict = {
#          #   'header': 
#             'data': result
#        }  
#        results.append(datadict)
#    #print (results)
#    return results
#        result=('\t%s - %.2f'%
#          (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()*100))
#        print(result)
#    return Video.from_file(vid,layout=Layout(width='80%', height='80%'))

#SlowFast('static/v_Basketball_g01_c01.mp4')
    

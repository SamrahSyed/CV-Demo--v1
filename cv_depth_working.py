import os
import argparse
import time
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth

import matplotlib as mpl
import matplotlib.cm as cm
import cv2

def depth(vid):
    # using cpu
    ctx = mx.cpu(0)
    
    import os, shutil
    folder = './dataForDepth'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
    
            
           
    # Opens the Video file
#    cap=vid
    cap= cv2.VideoCapture(vid)
#    cap= cv2.VideoCapture('abseiling_k400.mp4')
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('./dataForDepth/frame'+str(i)+'.jpg',frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()
    
    # data_path = os.path.expanduser("2011_09_26/2011_09_26_drive_0095_sync/image_02/data")
    data_path = os.path.expanduser("dataForDepth")
    
    files = os.listdir(os.path.expanduser(data_path))
    files.sort()
    
    raw_img_sequences = []
    for file in files:
        file = os.path.join(data_path, file)
        img = pil.open(file).convert('RGB')
        raw_img_sequences.append(img)
    #    result=raw_img_sequences.append(img)
    #    print(result)
        
    
    
    original_width, original_height = raw_img_sequences[0].size
    
    # img = pil.open(file).convert('RGB')
    
    raw_img_sequences.append(img)
    
    original_width, original_height = raw_img_sequences[0].size
    model_zoo = 'monodepth2_resnet18_kitti_mono_stereo_640x192'
    model = gluoncv.model_zoo.get_model(model_zoo, pretrained_base=False, ctx=ctx, pretrained=True)
    min_depth = 0.1
    max_depth = 100
    
    # while use stereo or mono+stereo model, we could get real depth value
    scale_factor = 5.4
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    
    feed_height = 192
    feed_width = 640
    
    pred_depth_sequences = []
    pred_disp_sequences = []
    for img in raw_img_sequences:
        img = img.resize((feed_width, feed_height), pil.LANCZOS)
        img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)
    
        outputs = model.predict(img)
        mx.nd.waitall()
        pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
        t = time.time()
        pred_disp = pred_disp.squeeze().as_in_context(mx.cpu()).asnumpy()
        pred_disp = cv2.resize(src=pred_disp, dsize=(original_width, original_height))
        pred_disp_sequences.append(pred_disp)
    
        pred_depth = 1 / pred_disp
        pred_depth *= scale_factor
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        pred_depth_sequences.append(pred_depth)
        
        
    output_path = os.path.join(os.path.expanduser("."), "tmp")
    
    pred_path = os.path.join(output_path, 'pred')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    
    for pred, file in zip(pred_depth_sequences, files):
        pred_out_file = os.path.join(pred_path, file)
        cv2.imwrite(pred_out_file, pred)
        
        
    rgb_path = os.path.join(output_path, 'rgb')
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    
    output_sequences = []
    for raw_img, pred, file in zip(raw_img_sequences, pred_disp_sequences, files):
        vmax = np.percentile(pred, 95)
        normalizer = mpl.colors.Normalize(vmin=pred.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(pred)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
    
        raw_img = np.array(raw_img)
        pred = np.array(im)
        output = np.concatenate((raw_img, pred), axis=0)
        output_sequences.append(output)
    
        pred_out_file = os.path.join(rgb_path, file)
        cv2.imwrite(pred_out_file, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

    width = int(output_sequences[0].shape[1] + 0.5)
    height = int(output_sequences[0].shape[0] + 0.5)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        os.path.join(output_path, 'demo1.mp4'), fourcc, 20.0, (width, height))  
    print(out)
"""     width = int(output_sequences[0].shape[1])
    height = int(output_sequences[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
#     out = cv2.VideoWriter(
#        os.path.join('static', 'depth_demo3.mp4'), fourcc, 20.0, (width, height))
     
   # out = cv2.VideoWriter('static/uploads', 'depth67.mp4')
    out = cv2.VideoWriter(
        os.path.join('static/uploads', 'depth67.mp4'), fourcc, 20.0, (width, height)) 
#    file.save(out)
    print(out) """

depth('./static/abseiling_k400.mp4')
#    return
"""     for frame in output_sequences:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
        out.write(frame)
        cv2.imshow('demo1', frame)
     #  return show_vid
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # uncomment to display the frames
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(2)
#     """
#depth('static/uploads/abseiling_k4001.mp4')
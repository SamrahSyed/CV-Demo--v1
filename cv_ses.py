import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg


def FCN (pic):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
    ctx = mx.cpu(0)
    img = image.imread(pic)
    plt.sca(ax[0])
    plt.imshow(img.asnumpy())
    img = test_transform(img, ctx)
    model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save('output.png')
    mmask = mpimg.imread('output.png')
    plt.sca(ax[1])
    plt.imshow(mmask)
    plt.savefig(pic, bbox_inches="tight")
    return
   # return #plt.show()

#FCN('static/uploads/aj1.jpeg')

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform

def PSPNet (pic):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
    ctx = mx.cpu(0)
    img = image.imread(pic)
    plt.sca(ax[0])
    plt.imshow(img.asnumpy())
    img = test_transform(img, ctx)
    model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save('output.png')
    mmask = mpimg.imread('output.png')
    plt.sca(ax[1])
    plt.imshow(mmask)
    plt.savefig(pic, bbox_inches="tight")
    return #plt.show()

#PSPNet('static/uploads/aj1.jpeg')

def DeepLabV3(pic):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
    ctx = mx.cpu(0)
    img = image.imread(pic)
    plt.sca(ax[0])
    plt.imshow(img.asnumpy())
    img = test_transform(img, ctx)
    model = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save('output.png')
    mmask = mpimg.imread('output.png')
    plt.sca(ax[1])
    plt.imshow(mmask)
    plt.savefig(pic, bbox_inches="tight")
    return #plt.show()

#DeepLabV3('static/uploads/2people.jpeg')
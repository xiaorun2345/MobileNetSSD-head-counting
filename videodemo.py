import numpy as np
import sys, os
import cv2
caffe_root = '/home/lenovo/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
net_file= '/home/lenovo/ssd/caffe/examples/MobileNet-SSD/project/MobileNetSSD_deploy.prototxt'  
caffe_model='/home/lenovo/ssd/caffe/examples/MobileNet-SSD/project/mobilenet_iter_20000.caffemodel'  
# caffe_model='mobilenet_iter_84000.caffemodel'
if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file, caffe_model, caffe.TEST) 
#CLASSES = ('background',
#        'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
##           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
 #          'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('background', 'person')
cap=cv2.VideoCapture('1.mov')
videoname='2.avi'
fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', '2')
out=cv2.VideoWriter(videoname,fourcc,10,(640,480))

def preprocess(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img * 0.007843
    return img
def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)
while 1:
    ret,frame=cap.read()
    if ret==False:
        print 'erro' 
    origimg=frame
    print frame.shape
    cv2.imshow("xi",frame)
    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)
    m = len(box)
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(origimg, p1, p2, (0, 0, 255), 5)
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
        cv2.putText(origimg, 'person: ', (60, 80), cv2.FONT_ITALIC, 2, (0, 0, 255), 3)
        cv2.putText(origimg, str(m), (300, 80), cv2.FONT_ITALIC, 2, (0, 0, 255), 3)
    cv2.imshow("SSD", origimg)
    out.write(origimg)
    cv2.imwrite("test.jpg",origimg) 
    print (origimg.shape)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break





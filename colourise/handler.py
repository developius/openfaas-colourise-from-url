import os
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import sys, time, warnings
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
import caffe, contextlib, io, tempfile, shutil, urlparse, requests

def download_file(url):
    r = requests.get(url, stream=True)
    u = urlparse.urlparse(url)
    dest = os.path.join(tempfile.gettempdir(), os.path.basename(u.path))
    with open(dest, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    return dest

"""
Input:
http://example.com/image.png

Output:
<binary>
"""
def handle(url):
    caffe.set_mode_cpu()

    # Select desired model
    net = caffe.Net('./models/colorization_deploy_v2.prototxt', './models/colorization_release_v2.caffemodel', caffe.TEST)

    (H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
    (H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

    pts_in_hull = np.load('./resources/pts_in_hull.npy') # load cluster centers
    net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        now = str(int(round(time.time() * 1000)))

        filename_out = now + '_output.jpg'
        file_path_out = os.path.join(tempfile.gettempdir(), filename_out)

        file_path_in = download_file(url)

        # load the original image
        img_rgb = caffe.io.load_image(file_path_in)

        start = time.time()

        img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
        img_l = img_lab[:,:,0] # pull out L channel
        (H_orig,W_orig) = img_rgb.shape[:2] # original image size

        # create grayscale version of image (just for displaying)
        img_lab_bw = img_lab.copy()
        img_lab_bw[:,:,1:] = 0
        img_rgb_bw = color.lab2rgb(img_lab_bw)

        # resize image to network input size
        img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
        img_lab_rs = color.rgb2lab(img_rs)
        img_l_rs = img_lab_rs[:,:,0]

        net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
        net.forward() # run network

        ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
        ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
        img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
        img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb

        plt.imsave(file_path_out, img_rgb_out)

        with open(file_path_out, 'rb') as f:
            print(f.read())

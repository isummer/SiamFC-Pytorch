import numpy as np
import cv2

def get_center(x):
    return (x - 1.) / 2.

def xyxy2cxcywh(bbox):
    return get_center(bbox[0]+bbox[2]), \
           get_center(bbox[1]+bbox[3]), \
           (bbox[2]-bbox[0]), \
           (bbox[3]-bbox[1])

def crop_and_pad(im, pos, model_sz, original_sz, avg_chans=None):
    '''
    # obtain image patch, padding with avg channel if area goes outside of border
    '''
    if avg_chans is None:
        avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]

    if original_sz is None:
        original_sz = model_sz

    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert (im_sz[0] > 2) & (im_sz[1] > 2), "The size of image is too small!"
    c = (sz + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[0] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = max(0, 1 - context_xmin)  # in python, index starts from 0
    top_pad = max(0, 1 - context_ymin)
    right_pad = max(0, context_xmax - im_sz[1])
    bottom_pad = max(0, context_ymax - im_sz[0])

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    im_R = im[:, :, 0]
    im_G = im[:, :, 1]
    im_B = im[:, :, 2]

    # padding
    if (top_pad != 0) | (bottom_pad != 0) | (left_pad != 0) | (right_pad != 0):
        im_R = np.pad(im_R, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant',
                      constant_values=avg_chans[0])
        im_G = np.pad(im_G, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant',
                      constant_values=avg_chans[1])
        im_B = np.pad(im_B, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), 'constant',
                      constant_values=avg_chans[2])

        im = np.stack((im_R, im_G, im_B), axis=2)

    im_patch_original = im[int(context_ymin) - 1:int(context_ymax), int(context_xmin) - 1:int(context_xmax), :]

    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch_original, (int(model_sz), int(model_sz)), interpolation=cv2.INTER_CUBIC)
    else:
        im_patch = im_patch_original

    return im_patch

def get_exemplar_image(img, bbox, size_z, context_amount, avg_chans=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    exemplar_img = crop_and_pad(img, np.array([cy, cx]), size_z, round(s_z), avg_chans)
    return exemplar_img, scale_z, s_z

def get_instance_image(img, bbox, size_z, size_x, context_amount, avg_chans=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x
    instance_img = crop_and_pad(img, np.array([cy, cx]), size_x, round(s_x), avg_chans)
    return instance_img, scale_x, s_x

def get_pyramid_instance_image(img, center, in_side_scaled, num_scales, size_x=255, avg_chans=None):
    if avg_chans is None:
        avg_chans = tuple(map(int, img.mean(axis=(0, 1))))

    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = size_x / float(min_target_side)

    search_side = int(round(beta * max_target_side))
    search_patch = crop_and_pad(img, np.array([center[1], center[0]]), search_side, round(max_target_side), avg_chans)

    # im_crop_list = []
    pyramid = []
    tmp_pos = np.array([(search_side+1)/2., (search_side+1)/2.])
    for s in range(num_scales):
        target_side = round(beta * in_side_scaled[s])
        im_crop_s = crop_and_pad(search_patch, tmp_pos, size_x, round(target_side), avg_chans)
        # im_crop_list.append(im_crop_s)
        pyramid.append(im_crop_s)

    return pyramid

"""
def get_pyramid_instance_image(img, center, size_x, size_x_scales, avg_chans):
    if avg_chans is None:
        avg_chans = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, np.array([center[1], center[0]]), size_x, round(size_x_scale), avg_chans)
            for size_x_scale in size_x_scales]
    return pyramid
"""

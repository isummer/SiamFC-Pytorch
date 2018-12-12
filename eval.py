import os
import cv2
import numpy as np
import json
import time
import argparse

from siamfc.tracker import SiamFCTracker
from siamfc.eval_otb import eval_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test siamfc on OTB')
    parser.add_argument('--dataset', default='OTB2013', choices=['OTB2013', 'OTB2015'], help='test on which dataset')
    parser.add_argument('--model', default='./models/siamfc_15.pth', help='path to trained model')
    args = parser.parse_args()

    dataset = args.dataset                                          # OTB2013 or OTB2015
    base_path = os.path.join('dataset', dataset)                    # path to OTB2015/OTB2013 (image path)
    json_path = os.path.join('dataset', dataset + '.json')          # path to OTB2015.json/OTB2013.json (annotations)
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())                                 # video name
    
    visualization = True # True

    tracker = SiamFCTracker(args.model)
    
    # loop videos
    for video_id, video in enumerate(videos):
        video_name = annos[video]['name']                           # tracked video
        imgs_dir = os.path.join(base_path, video_name, 'img')
        imgs_list = [os.path.join(imgs_dir, im_f) for im_f in sorted(annos[video]['image_files'])]   # path to tracked frames
        gt = np.array(annos[video]['gt_rect']).astype(np.float)     # groundtruth of tracked video
        num_frames = len(imgs_list)                                   # number of tracked frames
        assert num_frames == len(gt)

        bboxes = np.zeros((num_frames, 4))

        im = cv2.cvtColor(cv2.imread(imgs_list[0]), cv2.COLOR_BGR2RGB)
        init_rect = np.array(annos[video]['init_rect']).astype(np.float)
        x, y, w, h = init_rect
        bboxes[0, :] = x, y, w, h
        
        tracker.init(im, init_rect)

        tic = time.time()
        for f in range(1, len(imgs_list)):
            im = cv2.cvtColor(cv2.imread(imgs_list[f]), cv2.COLOR_BGR2RGB)
            rect = tracker.update(im)
            xmin, ymin, xmax, ymax = [int(x) for x in rect]
            bboxes[f, :] = xmin, ymin, xmax-xmin+1, ymax-ymin+1

            if visualization:
                im_to_show = im.copy()
                cv2.rectangle(im_to_show, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              (0, 255, 0), 3)
                cv2.putText(im_to_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                im_to_show = cv2.resize(im_to_show, None, None, fx=0.25, fy=0.25)
                cv2.imshow(video, im_to_show)
                cv2.waitKey(1)
        cv2.destroyAllWindows()

        toc = time.time() - tic
        fps = num_frames / toc
        print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.1f}fps'.format(video_id, video, toc, fps))

        # save result
        test_path = os.path.join('result', dataset, 'Siamese-fc_test')
        if not os.path.isdir(test_path): os.makedirs(test_path)
        result_path = os.path.join(test_path, video + '.txt')
        with open(result_path, 'w') as f:
            for x in bboxes:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')
         
    eval_auc(dataset, 'Siamese-fc_test', 0, 1)


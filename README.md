## SiamFC-PyTorch

* This is the PyTorch 0.4 implementation of SiamFC tracker [1], which was originally <a href="https://github.com/bertinetto/siamese-fc">implemented</a> using MatConvNet [2].
* Support multi-gpu training

## Goal

* A more compact implementation of SiamFC [1].
* Reproduce the results of SiamFC [1].

## Requirements

* Python 3.6
* Python-opencv
* PyTorch 0.4.0

## Data curation 

* Download <a href="http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz">ILSVRC15</a>, and unzip it (let's assume that `$ILSVRC2015_Root` is the path to your ILSVRC2015)
* run `./script/careate_dataset.sh`, then you will get two json files `imdb_video_train.json` (~ 430MB) and `imdb_video_val.json` (~ 28MB) in current folder, which are used for training and validation.

## Train

* run `./script/train.sh`
* **some notes for training:**
  * the options for training are in `Config.py`
  * each epoch (50 in total) may take 6 minuts (Nvidia Titan Pascal, num_worker=8 in my case)

## Tracking

* Take a look at `Config.py` first, which contains all parameters for tracking
* Change `self.net_base_path` to the path saving your trained models
* Change `self.net` to indicate whcih model you want for evaluation, and I've uploaded a trained model `SiamFC_45_model.pth` in this rep (located in $SiamFC-PyTorch/Train/model/)
* The default parameters I use for my results is as listed in `Config.py`.
* run `./script/demo.sh`

## References

[1] L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, and P. H. Torr. Fully-convolutional siamese networks for object tracking. In ECCV Workshop, 2016.

[2] A. Vedaldi and K. Lenc. Matconvnet – convolutional neural networks for matlab. In ACM MM, 2015.

[3]https://github.com/StrangerZhang/SiamFC-PyTorch

[3]https://github.com/HengLan/SiamFC-PyTorch



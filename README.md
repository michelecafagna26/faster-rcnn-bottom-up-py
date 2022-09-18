# Bottom-up Attention with Detectron2 (Pytorch)

Extract features and bounding boxes predictions in a few lines of Python code.

This repo is a cleaned version of [airsplay] (https://github.com/airsplay/py-bottom-up-attention) repo. For more details refer to that repo.

The detectron2 system with **exactly the same model and weight** as the Caffe VG Faster R-CNN provided in [bottom-up-attetion](https://github.com/peteanderson80/bottom-up-attention).

## Installation

```bash
git clone https://github.com/michelecafagna26/faster-rcnn-bottom-up-py.git
cd faster-rcnn-bottom-up-py

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```

### Quick start: Feature Extraction + Object Detection 


```python
from wrappers import FasterRCNNBottomUp
import cv2

IMG_FILE = "data/images/COCO_train2014_000000084002.jpg"

cfg_file = "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool_wrapper.yaml"
vg_objects = "data/genome/1600-400-20/objects_vocab.txt"

im = cv2.imread(IMG_FILE)
model = FasterRCNNBottomUp(cfg_file, object_txt = vg_objects, MAX_BOXES=150, MIN_BOXES=150)

instances, boxes = model([im], return_features=True)

```
To access the predicted object class:

```python
class_id = instances[0].pred_classes
model.classes['thing_classes'][class_id]

```



## Note from the original repo
1. The default weight is same to the 'alternative pretrained model' in the original github [here](https://github.com/peteanderson80/bottom-up-attention#demo), which is trained with 36 bbxes. If you want to use the original detetion trained with 10~100 bbxes, please use the following weight:
   ```
   http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl
   ```
2. The coordinate generated from the code is (x_left_corner, y_top_corner, x_right_corner, y_bottom_corner). Here is a visualization. Suppose the `box = [x0, y0, x1, y1]`, it annotates an RoI of:
   ```
   0-------------------------------------
    |                                   |
    y0 box[1]   |-----------|           |
    |           |           |           |
    |           |  Object   |           |
    y1 box[3]   |-----------|           |
    |                                   |
   H----------x0 box[0]-----x1 box[2]----
    0                                   W
   ```
3. If the link breaks, please use this Google Drive: https://drive.google.com/drive/folders/1ICBed8F9JaayAshptGMiGtRj78esg3m4?usp=sharing.

## External Links
1. The orignal CAFFE implementation [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), and its [docker image](https://hub.docker.com/r/airsplay/bottom-up-attention).
2. [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) maintained by [MIL-LAB](http://mil.hdu.edu.cn/). 



## Acknowledgement
- original repo [airsplay]()
- The Caffe2PyTorch conversion code (not released here) is based on [Ruotian Luo](https://ttic.uchicago.edu/~rluo/)'s [PyTorch-ResNet](https://github.com/ruotianluo/pytorch-resnet) project. 
- The project also refers to [Ross Girshick](https://www.rossgirshick.info/)'s old [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) on its way.


## References

Detectron2:
```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

Bottom-up Attention:
```BibTeX
@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}
```


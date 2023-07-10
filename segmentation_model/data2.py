import os      
import cityscapesscripts

os.environ['CITYSCAPES_DATASET'] = '/home/inesh/projects/stickerbomb/segmentation_model/dataset'
cityscapesscripts.csCreateTrainIdLabelImgs()



---   
<div align="center">
 
# Fire detection using Detectron2

</div>
 
## Description
This repository holds the implementation of a fire detection system using Detectron2.
We use small fire dataset in Pascal VOC data format which only has one class: fire.
We'll train a fire detection model from an existing model pre-trained on COCO dataset, available in detectron2's model zoo.
Model in our case is Faster R-CNN and ResNet+FPN backbone with standard conv and FC heads for mask and box prediction, respectively. 
It obtains the best speed/accuracy tradeoff.

![](demo.gif)

## Getting Started   
The easiest way to repeat the whole experiment is to run [workflow.ipynb](workflow.ipynb]) in Google Colab.

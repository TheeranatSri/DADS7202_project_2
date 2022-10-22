# <p align="center"> One Piece Character Detection </p>

by Capybarista Team
<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Header.jpg"> </span>

# Definition
- **Model 1** : Roboflow (Default) - (0.7)<br>
  Yolov5l on app.roboflow.com
- **Model 2** : Yolov5l - Weight False, Freeze False<br>
  Yolov5l unfreeze all layers and don't use pretrain weights.
- **Model 3** : Yolov5l - Weight True, Freeze False<br>
  Yolov5l unfreeze all layers and  use pretrain weights.
- **Model 4** : Yolov5l - Weight False, Freeze True<br>
  Yolov5l freeze all layers and don't use pretrain weights.
- **Model 5** : Yolov5l - Weight True, Freeze True<br>
  Yolov5l freeze all layers and  use pretrain weights.
- **Model 6** : Yolov5l - Weight True, Freeze 10<br>
  Yolov5l freeze 10 layers (backbone) and use pretrain weights.



# Brief Result:

- Model comparison with mAP@0.5 score and Running time
   - Model 1 (Default) - (0.7 / not available)
   - Model 2 (0.709 / 5.36 hr)
   - Model 3 (0.787 / 5.35 hr)
   - Model 4 (0.002 / 2.42 hr)
   - Model 5 (0.066 / 2.43 hr)
   - Model 6 (0.673 / 3.22 hr)

- The Best Model (Highest score) is Model 3 (Unfreeze all layer and use pretrain weights).
- Alternative model (Balance between score and running time) is Model 6.
  
# Introduction:
- Object detection is a computer vision technique that allows us to recognize and locate objects in images or videos. Object detection can be used to count objects in a scene and determine and track their precise locations, all while accurately labeling them, using this type of identification and localization. 
Below is a comparison of image classification, object detection, and instance segmentation.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Picture1.png"> </span>

<br>

## YOLOv5
  
- YOLOv5: YOLO, which stands for "You Only Look Once," is an object detection algorithm that divides images into grids. Each grid cell is in charge of detecting objects within itself. Because of its speed and accuracy, YOLO is one of the most well-known object detection algorithms.
[[Li, Karpathy,Johnson]](http://cs231n.stanford.edu/2016/)
- The YOLOv5 algorithm design follows the consistent idea of the YOLO series: the image to be detected was processed through an input layer (input) and sent to the backbone for feature extraction, which has convolution layers. Then, feature maps were generated to detect the objects in the image. Following that, the prediction head (head) will be notified, and the confidence of bounding-box were executed for each pixel in the feature map to obtain a multi-dimensional array (BBoxes) containing object class, class confidence, box coordinates, width, and height information. To filter the useless information in the array, and performing a non-maximum suppression (NMS) process (which selects the best bounding box for an object and rejects or "suppresses" all other bounding boxes). The inference process refers to the process of converting the input image into BBoxes [[Haiying Liu Fengqian Sun,Jason Gu,Lixia Deng]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9371183/)

<p align="center" width="100%"><img width="80%" src="pic/architecture_yolov5.jpg"> </span><br>
Inference process of YOLOv5


## Roboflow
- Roboflow is a computer vision platform that enables users to build computer vision models faster and more accurately by providing improved data collection, preprocessing, and model training techniques. Roboflow allows users to upload custom datasets, draw annotations, change image orientations, resize images, change image contrast, and perform data augmentation. 
It can also be used to train models.<br>
*Link for more detail*: https://docs.ultralytics.com/ , https://github.com/ultralytics/yolov5 , https://pypi.org/project/yolov5/

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow.jpg"> </span>

## Image Augmentation
- Image augmentation is an efficacious technique when we don’t have an ample amount of data for training a deep learning model. Our team uses Roboflow to enhance images (due to time limitation, we set criteria of image augmentation as default of Roboflow). The image augmentation options in Roboflow are depicted in the figure below.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/augmentation-options.png"> </span>

## mAP
- The mean Average Precision, or mAP score, is calculated by averaging the AP across all classes and/or the overall IoU thresholds, depending on the detection challenges.[[Shivy Yohanandan]](https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2)

## IoU
- Intersection over Union is the defacto evaluation metric used in object
detection. It's used to determine which predictions are true and which are false. An accuracy threshold must be chosen when using IoU as an evaluation metric. [[Rezatofighi,Tsoi,Gwa,etc.]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf)
<p align="center" width="20%">
    <img width="35%" src="http://ronny.rest/media/tutorials/localization/ZZZ_IMAGES_DIR/iou_formula.png"> </span>


Step of Roboflow for One Piece Character Detection Project
1.	Upload photo into Roboflow
2.	Label One Piece Character in all of the photos in the datasets.
3.	Image preprocessing & Image augmentation
4.	Train model.
5.	Deploy.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow%20step.png"> </span>

Image source and detail of roboflow tutorial --> https://blog.streamlit.io/how-to-use-roboflow-and-streamlit-to-visualize-object-detection-output/

# Our project in Roboflow:
- Project Link: https://app.roboflow.com/dl-yjboe/dads7202_hw2
- Total number of images = 1182 (The number of photos for each character is listed below.) Some images have multiple characters)
- Total number of photos after image augmentation = 11232
- Train-Test Split are 70% : 20% : 10% 
  - After Image Augmentation, Train : Validation : Test are 87% : 8% : 5%
- Other settings are shown below

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Number%20of%20character.jpg"> </span>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow%20setting.jpg"> </span>


# Roboflow's outcome:
- mAP@0.5 for Train / Validation / Test of all character  = 71% / 68% / 70%

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow%20score.png"> </span>

However, Roboflow have tuning limitation. Next step, team will use jupyter in colab to adjust hyperparameter.

---------------------------------------------------
# Our project in Jupyter-Colab:
- Project Link: 
   - https://colab.research.google.com/drive/1hfHihyPVFt18axpft3brB-uG-jAQ9OoM?usp=sharing#scrollTo=ii8qC1HDUzZ6

- Six models were compared based on batch size 16, 100 epochs, and the default hyperparameter.

# mAP@0.5 score from Jupyter-Colab:
-    **Model 1** mAP@0.5  0.71
<p align="center" width="100%">
    <img width="80%" src="pic/model1_result.png"> </span><br>

   - **Model 2** mAP@0.5  0.709
<p align="center" width="100%">
    <img width="80%" src="pic/yolov5l_W_False_F_False_PR_curve.png"> </span>

   - **Model 3** mAP@0.5  0.787

<p align="center" width="100%">
    <img width="80%" src="pic/yolov5l_W_True_F_False_PR_curve.png"> </span>

   - **Model 4** mAP@0.5 0.002

<p align="center" width="100%">
    <img width="80%" src="pic/yolov5l_W_False_F_True_PR_curve.png"> </span>

   - **Model 5** mAP@0.5  0.066

<p align="center" width="100%">
    <img width="80%" src="pic/yolov5l_W_True_F_True_PR_curve.png"> </span>

   - **Model 6** mAP@0.5  0.673 

<p align="center" width="100%">
    <img width="80%" src="pic/yolov5l_W_True_F10_True_PR_curve.png"> </span>



# Jupyter-Colab's outcome:
- The best score model = Yolov5l - Weight True, Freeze False at 0.787
- However, running time is high. The running time of each model is shown below

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Running%20time.png"> </span>

- Alternative model (Balance between score and running time) = Yolov5l - Weight True Freeze 10 
   - Score drop = ~15%
   - Save time = ~40%
- Observation: 
   - Score: unfreeze layer model  > freeze layer model
   - Score: allow to update weight model  > No update weight model
   - Running time: unfreeze layer model  > freeze layer model
   - Running time: allow to update weight model  ~ No update weight model

From the six training models, we can summarize as follows;

In terms of mAP@0.5 score, the model with the best mAP@0.5 score is model number 2, which is an unfreeze model that uses the weight from the pre-train model. When compared to model number 1, which is an unfreeze model that does not use the weight from the pre-train model, the mAP@0.5 result is significantly different even though the training time used is not significantly different.

In terms of training time, although model number 5 and model number 1 where the mAP@0.5 score are not significantly different, however the training time of model number 5 is significantly less than model number 1.

Furthermore, models 3 and 4, which are trained by freezing the entire model except the inference layers, produce irrelevant results because the Head and Neck sections, which are not part of Feature Extraction, do not change the back propagation in relation to the prediction in accordance with the new dataset.

# Next step:
- Use more features of image augmentation (Cutout, Grayscale, ...)
- Comparison to other versions of YoloV5 (YoloV5m,  YoloV5x, ...)
- Modify the labeling standard; for example, some images may only show the character's side or ears, making it difficult to detect.
- Unfreeze the parts of the head and neck separately to compare the results of the unfreezes.
- Check whether the results of models 1,2,5 and yoloV5 versions s,m, and x are the same.
- Increase the number of images in the dataset to see if it improves prediction accuracy.


# References:
- Object Detection:
  - https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852
  - https://machinelearningmastery.com/object-recognition-with-deep-learning/
  - https://medium.com/ml-research-lab/what-is-object-detection-51f9d872ece7
- Yolov5: 
  - https://docs.ultralytics.com/
  - https://github.com/ultralytics/yolov5
  - https://pypi.org/project/yolov5/
- Roboflow:
  - https://roboflow.com/
  - https://blog.streamlit.io/how-to-use-roboflow-and-streamlit-to-visualize-object-detection-output/
- mAP Definition
  - https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2
  - https://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf

---------------------------------------------------

# Capybarista Team Members:
- Theeranat Sringamdee 641042014 
- Patcharapruek Watanangura_6410412007 
- Nattapong Thanngam_6310422089 
- สุกิจ วาณิชฤดี_6310422092 
- On Minteer_6410414001 

<p align="center" width="100%">
    <img width="50%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Capybarista3.jpg"> </span>

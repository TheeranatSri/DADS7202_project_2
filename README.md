# <p align="center"> One Piece Character Detection </p>

by Capybarista Team
<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Header.jpg"> </span>



# Brief Result:

- Model comparison with mAP@0.5 score and Running time
   - roboflow (Default) - (0.7)
   - Yolov5l - Weight False Freeze False (0.709 / 5.36 hr)
   - Yolov5l - Weight True Freeze False (0.787 / 5.35 hr)
   - Yolov5l - Weight False Freeze True (0.002 / 2.42 hr)
   - Yolov5l - Weight True Freeze True (0.066 / 2.43 hr)
   - Yolov5l - Weight True Freeze 10 (0.673 / 3.22 hr)
   - Yolov5s - Weight True Freeze True (ยังไม่เสร็จ)

- The Best Model (Highest score) = Yolov5l - Weight True Freeze False (Unfreeze Model + Adjust weight from pre-train result)
- Alternative model (Balance between score and running time) = Yolov5l - Weight True Freeze 10

# Introduction:
- Object detection is a computer vision technique that allows us to recognize and locate objects in images or videos. Object detection can be used to count objects in a scene and determine and track their precise locations, all while accurately labeling them, using this type of identification and localization. 
Below is a comparison of image classification, object detection, and instance segmentation.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/Picture1.png"> </span>

from: Standford University 2016 winter lectures CS231n Fei-Fei Li & Andrej Karpathy & Justin Johnson

- Roboflow is a computer vision platform that enables users to build computer vision models faster and more accurately by providing improved data collection, preprocessing, and model training techniques. Roboflow allows users to upload custom datasets, draw annotations, change image orientations, resize images, change image contrast, and perform data augmentation. It can also be used to train models.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow.jpg"> </span>

Image augmentation is an efficacious technique when we don’t have an ample amount of data for training a deep learning model. Our team uses Roboflow to enhance images (due to time limitation, we set criteria of image augmentation as default of Roboflow). The image augmentation options in Roboflow are depicted in the figure below.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/augmentation-options.png"> </span>
     
Step of Roboflow for One Piece Character Detection Project
1.	Upload photo into Roboflow
2.	Label One Piece Character in all of the photos in the datasets.
3.	Image preprocessing & Image augmentation
4.	Train model.
5.	Deploy.

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow%20step.png"> </span>

Image source and detail of Roboflow tutorial --> https://blog.streamlit.io/how-to-use-roboflow-and-streamlit-to-visualize-object-detection-output/

# Roboflow's outcome:
- Project Link: https://app.roboflow.com/dl-yjboe/dads7202_hw2/13
- mAP for Train / Validation / Test of all charactor = 71% / 68% / 70%
- Observation: Usopp is lowest score --> Usopp have 2 version (wear mask/no mask) but our team label only 1 label (ยังไม่เสร็จ)

<p align="center" width="100%">
    <img width="80%" src="https://github.com/NattapongTH/DADS7202_project_2/blob/main/pic/roboflow%20score.png"> </span>







---------------------------------------------------
Respone type vs K-MEAN result 

![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/5.%20KMEAN%20vs%20Respone.JPG)

Respone type vs key parameter

![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/6.JPG)
![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/7.JPG)

- Create new parameter to use in prediction model that is shown in following

![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/3.%20Parameter.JPG)

- Feature selection
                             
![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/4.%20Feature%20selection.JPG)

- Highest score

![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/8.JPG)
![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/9.JPG)

- Conclusion

![alt](https://github.com/NattapongTH/NattapongTH-6310422089_BADS7105/blob/main/Homework%2008%20%E2%80%93%20Campaign%20Response%20Model/Photo/11.jpg)

# Capybarista Group <p> *Object Detection One Piece* </p>

# 1. Overview 
Data  จำนวน 1182 รูป ประกอบก้วย 9 คลาสดังนี้
- Brook         92  รูป
- Chopper       204 รูป
- Franky        175 รูป
- Luffy         364 รูป
- Nami          205 รูป
- Robin         217 รูป
- Sanji         171 รูป
- Usopp         271 รูป
- Zoro          263 รูป
  
แบ่งเป็น Train:Test:Split   70 : 20 : 10

กระบวนการ Create Dataset ด้วย Roboflow มีดังนี้
1. PREPROCESSING
    - Auto-Orient: Applied
    - Resize: Stretch to 416xv416
    - Tile: 2 rows x 2 columns
2. AUGMENTATIONS
    - Outputs per training example: 3
    - Flip: Horizontal, Vertical
    - Crop: 0% Minimum Zoom, 20% Maximum Zoom
    - Rotation: Between -31° and +31°
    - Brightness: Between -25% and +25%
    - Blur: Up to 2px
  
รูปภาพที่ผ่านการ AUGMENTATIONS มีจำนวนทั้งสิ้น 11232 รูป แบ่งได้เป็น
1. Train      จำนวน 9756 รูป ( 87% )
2. Validation จำนวน 984  รูป (  8% )
3. Test       จำนวน 528  รูป (  5% ) 

Model ที่ใช้ในการทดสอบคือ Yolov5s และ Yolov5l
แบ่งการทดสอบดังนี้
1. Yolov5l unfreeze model และให้ Model ทำการปรับ Weight เอง
2. Yolov5l unfreeze model และใช้ weight ที่ผ่าน pretrain (yolov5l.pt)
3. Yolov5l Freeze   model ทั้งหมดยกเว้น Layer สุดท้ายสำหรับใช้ Inference และใช้ weight ที่ผ่าน pretrain (yolov5l.pt)
4. Yolov5l Freeze  model ทั้งหมดยกเว้น Layer สุดท้ายสำหรับใช้ Inference และให้ Model ทำการปรับ Weight เอง
5. Yolov5l Freeze  10 layer แรกและใช้ weight ที่ผ่าน pretrain  
6. Yolov5s Freeze  model ทั้งหมดยกเว้น Layer สุดท้ายสำหรับใช้ Inference และใช้ weight ที่ผ่าน pretrain  (yolov5s.pt)


# 2. What is Yolov5?
The object detection method YOLO, which stands for "You Only Look Once," divide images into a grid structure. In the grid, each cell is in charge of finding objects within of it. Due to its accuracy and speed, YOLO is one of the most well-known object detection algorithms.

# 3. Code Detail

# 4. Result
## PR Curve
1. Yolov5l unfreeze และไม่ใช้ weight ที่ผ่าน pretrain <br>
<img src="pic/yolov5l_W_False_F_False_PR_curve.png" alt="drawing" align ="center" width ="400">

2. Yolov5l unfreeze model และใช้ weight ที่ผ่าน pretrain  <br>
<img src="pic/yolov5l_W_True_F_False_PR_curve.png" alt="drawing" align ="center" width ="400">

3. Yolov5l Freeze  model ทั้งหมดยกเว้น Layer สุดท้ายสำหรับใช้ Inference และใช้ weight ที่ผ่าน pretrain  
<img src="pic/yolov5l_W_True_F_True_PR_curve.png" alt="drawing" align ="center" width ="400">

4. Yolov5l Freeze  model ทั้งหมดยกเว้น Layer สุดท้ายสำหรับใช้ Inference และไม่ใช้ weight ที่ผ่าน pretrain <br>
<img src="pic/yolov5l_W_False_F_True_PR_curve.png" alt="drawing" align ="center" width ="400">   

5. Yolov5l Freeze  backbone 10 layer แรกและไม่ใช้ weight ที่ผ่าน pretrain  
<img src="pic/yolov5l_W_True_F10_True_PR_curve.png" alt="drawing" align ="center" width ="400">   

6. Yolov5s Freeze  model ทั้งหมดยกเว้น Layer สุดท้ายสำหรับใช้ Inferenceและใช้ weight ที่ผ่าน pretrain 



## mAP@0.5 ของโมเดลรูปแบบต่าง ๆ<br>
<img src="pic/chart_PR_curve.png " alt="drawing" align ="center" width ="400">  <br> 

โมเดลที่ให้ผลลัพธ์ดีที่สุด 3 อันดับแรกคือ โมเดลที่ 2 3 และ 5 มีค่า mAP@0.5 อยู่ที่ 0.787 0.709 และ 0.ตามลำดับ ส่วนโมเดล 3 และ 4 ที่มีการ Freeze ทั้งโมเดลจะให้ผลลัพธ์ที่ตรงข้ามกันอย่างสิ้นเชิง ซึ่งค่า mAP@0.5 ที่ได้ในแต่ละโมเดลมีความสอดคล้องกับ Precision-Recall Curve โดยโมเดลที่มีค่า mAP@0.5 มากเส้นโค้ง Precision-Recall ก็จะมาค่าเข้าใกล้ 1 เมื่อพิจารณา mAP@0.5 ในแต่ละ Epoch ของโมเดลที่ 1 2 และ 5 จะพบว่าค่า mAP@0.5 จะมีอัตราการเปลี่ยนแปลงที่ไม่แตกต่างจากเดิมมากนักตั้งแต่ Epoch 80 เป็นต้นไป ในส่วนของเวลาที่ใช้ในการประมวลผลของแต่ละโมเดลจะเป็นดังต่อไปนี้<br>

<img src="pic/cls_log_loss.png " alt="drawing" align ="center" width ="400">  <br> 

โมเดลที่ใช้เวลานานที่สุดคือ โมเดลที่ 1 2 5 ซึ่งใช้เวลา 5.355 5.353 และ 3.222 ชั่วโมงตามลำดับ


# 5. Summary
From the six training models, we can summarize as follows;

In terms of mAP@0.5 score, the model with the best mAP@0.5 score is model number 2, which is an unfreeze model that uses the weight from the pre-train model. When compared to model number 1, which is an unfreeze model that does not use the weight from the pre-train model, the mAP@0.5 result is significantly different even though the training time used is not significantly different.

In terms of training time, although model number 5 and model number 1 where the mAP@0.5 score are not significantly different, however the training time of model number 5 is significantly less than model number 1.

Furthermore, models 3 and 4, which are trained by freezing the entire model except the inference layers, produce irrelevant results because the Head and Neck sections, which are not part of Feature Extraction, do not change the back propagation in relation to the prediction in accordance with the new dataset.

# 6. Future Work
- Modify the labeling standard; for example, some images may only show the character's side or ears, making it difficult to detect.
- Unfreeze the parts of the head and neck separately to compare the results of the unfreezes.
- Check whether the results of models 1,2,5 and yoloV5 versions s,m, and x are the same.
- Increase the number of images in the dataset to see if it improves prediction accuracy.

# Reference

[Yolov5]([https://](https://github.com/ultralytics/yolov5))

[C3 model ]([https://](https://arxiv.org/abs/1812.04920))

# Open-Source-Models
Open-Source-Models is a archive for all the open source computer vision models. Training Computer Vision models is an arduous task which involves a series of strenuous tasks such as collecting the images, annotating them, uploading them on cloud(In case you don't have a rig with a beffey GPU) and training them for hours and hours (which also requires you to find a workaround so that the colab doesn't timeout). All the steps mentioned above are to be executed without making any error as a small oversight can lead to a model trained with faulty config file, incorrect annotations etc. Thanks to all the generous people in the field of computer vision which are doing all the above tasks and providing thier work to others as an open source project, so that not everyone has to reinvent neural networks and can focus on the actual task that has to be carried out with the model.

This archive consists of models with different architecture, accuracy, and framework in the same category as different use cases demand different types of model to achieve similar goals.

## Contribution
This project cannot work without YOUR help. Everyone is encouraged to contribute to this project by listing the models they have trained after spending endless time and efforts to train them, so that everyone in the community is aware about its existance and can use them for their purpose.

### Format for contribution

* Add the source links(blogs, github repo) to the model and all the supporting files in the respective category along with information such as, number of classe(s), name of classe(s), number of images used for training, type of network(detection, segmentation, classification) and if possible also include a performance metric.

* You can also contribute to this project even if you haven't trained a model yet by testing the models already listed here and test it for performance, accuracy and report if the link is broken or the the file does not exist on the mentioned link.


## Can't find the model here?
Incase, you aren't able to find a model in here that fits your requirement and planning to train your own model, You can checkout the Google [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html). Here you can find annotated images that can be downloaded as per your convinience with [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit) and use this [fork](https://github.com/theAIGuysCode/OIDv4_ToolKit) in case you want the annotations that can be used to train a [YOLO](https://github.com/AlexeyAB/darknet) model.

If you cannot train a model for some reason, you can put up a request in the [issues](https://github.com/shubham-shahh/Open-Source-Models/issues) and see if someone can help you with that.


## Models Archive

### License plate detector
This section consists of link to models that has **License plate** or **number plate** as one of their classes.

1. License plate detector
   * **Model Architecture -** YOLOv4 
   * **Dataset-** [Open Images Dataset](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false&c=%2Fm%2F01jfm_)
   * **Number of training examples -** 1500
   * **Accuracy Metric -** (mAP@0.50) = 88.57%
   * **Number of classe(s) -** 1
   * **Link to the model and supporting files -** [Model](https://github.com/theAIGuysCode/yolov4-custom-functions)
   * **Author Remarks -** NA

2. License plate detector + Character detection
   * **Model Architecture -** YOLO 
   * **Dataset-** [Dataset](https://github.com/TheophileBuy/LicensePlateRecognition)
   * **Number of training examples -** 1900
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** 1
   * **Link to the model and supporting files -** [Model](https://github.com/TheophileBuy/LicensePlateRecognition)
   * **Author Remarks -** NA

3. License plate detector along with type of the vehicle
   * **Model Architecture -** YOLOv3-tiny
   * **Dataset-** NA
   * **Number of training examples -** 1700+
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** 10
   * **Link to the model and supporting files -** [Model](https://github.com/SumanSudhir/Vehicle-and-Its-License-Plate-detection)
   * **Author Remarks -** NA

4. License plate detector
   * **Model Architecture -** YOLOv3 
   * **Dataset-** NA
   * **Number of training examples -** NA
   * **Accuracy Metric -** (mAP@0.50) = NA
   * **Number of classe(s) -** 1
   * **Link to the model and supporting files -** [Model](https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector)
   * **Author Remarks -** NA

5. License plate detector
   * **Model Architecture -** YOLOv3 
   * **Dataset-** [Dataset avaliable for academic use only](https://github.com/alitourani/yolo-license-plate-detection)
   * **Number of training examples -** 3000+
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** 1
   * **Link to the model and supporting files -** [Model](https://github.com/alitourani/yolo-license-plate-detection)
   * **Author Remarks -** NA


### Fire detector
This section consists of link to models that has **fire** as one of their classes.

1. Fire detector
   * **Model Architecture -** YOLOv3 
   * **Dataset-** Open Images Dataset
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** 1
   * **Link to the model and supporting files -** [Model](https://github.com/snehitvaddi/YOLOv3-Cloud-Based-Fire-Detection)
   * **Author Remarks -** NA

2. Fire and fire arms detector
   * **Model Architecture -** YOLOv3 
   * **Dataset-** NA
   * **Number of training examples -** NA
   * **Accuracy Metric -** [details in paper](https://github.com/atulyakumar97/fire-and-gun-detection/blob/master/ICESC2020_Published%20Paper.pdf)
   * **Number of classe(s) -** 2
   * **Link to the model and supporting files -** [Model](https://github.com/atulyakumar97/fire-and-gun-detection)
   * **Author Remarks -** NA

3. Fire detector
   * **Model Architecture -** YOLOv3 
   * **Dataset-** [FireNet](https://github.com/OlafenwaMoses/FireNET)
   * **Number of training examples -** 502 spilt into 2 parts, 412 for training 90 for validation
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/OlafenwaMoses/FireNET)
   * **Author Remarks -** NA

4. Fire and smoke detector
   * **Model Architecture -** YOLOv4 
   * **Dataset-** NA
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/gengyanlei/fire-detect-yolov4/blob/master/README_EN.md)
   * **Author Remarks -** NA

5. Fire detector
   * **Model Architecture -** InceptionV4-OnFire 
   * **Dataset-** [Durham Collections - Dunnings/Breckon, 2018](https://github.com/tobybreckon/fire-detection-cnn)
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/tobybreckon/fire-detection-cnn)
   * **Author Remarks -** NA


### Face detector
This section consists of link to models that has **face** as one of their classes.

1. YOLO Face
   * **Model Architecture -** YOLOv3 
   * **Dataset-** [WIDER FACE: A Face Detection Benchmark](https://github.com/sthanhng/yoloface)
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/sthanhng/yoloface)
   * **Author Remarks -** NA

2. YOLO Face Keras
   * **Model Architecture -** YOLOv3 
   * **Dataset-** [WIDER FACE: A Face Detection Benchmark](https://github.com/swdev1202/keras-yolo3-facedetection)
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/swdev1202/keras-yolo3-facedetection)
   * **Author Remarks -** NA 

3. Face detector
   * **Model Architecture -** YOLOv2-tiny 
   * **Dataset-** [WIDER FACE: A Face Detection Benchmark](https://github.com/zlmo/Face-Detection)
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/zlmo/Face-Detection)
   * **Author Remarks -** NA 

4. Face detector
   * **Model Architecture -** YOLOv2 
   * **Dataset-** FDDB+Dlib
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/azmathmoosa/azFace)
   * **Author Remarks -** NA 

5. Ultra light face detector
   * **Model Architecture -** NA 
   * **Dataset-** WIDER FACE: A Face Detection Benchmark
   * **Number of training examples -** NA
   * **Accuracy Metric -** NA
   * **Number of classe(s) -** NA
   * **Link to the model and supporting files -** [Model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
   * **Author Remarks -** NA 














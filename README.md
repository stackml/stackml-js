# StackML
**StackML is a simple GUI tool for non-AI people to use machine learning in browser.**

**This github repository hosts StackML Javascript library which enables you to use machine learning models in your web app with few lines of code.**

**StackML provides two main functionalities**
**1) Use pre-trained models**
**2) Train a model on your own dataset**


### Use pre-trained models
![](https://d3l5ba5b56ptip.cloudfront.net/website/images/models_2.png)


### Train a model on your own dataset
![](https://d3l5ba5b56ptip.cloudfront.net/website/images/train_2.png)

## [Live Demo](https://stackml.com/#section-live-demo)


## [Full Documentation](https://stackml.com/docs/getting-started/)

**Table of Contents**
* [Models Examples](#Examples)
* [Getting Started](#Getting-Started)
* [How to Use](#How-To-Use)
    *  [Pre-trained models](#Pre-trained-models)
        * [Image classification](#Image-Classification)
        * [YOLO](#YOLO)
        * [PoseNet](#PoseNet)
        * [BodyPix](#BodyPix)
        * [Face Detection](#Face-Detection)
        * [Face Landmark Detection](#Face-Landmark-Detection)
        * [Face Expression Recognition](#Face-Expression-Recognition)
    * [User trained models](#User-trained-models)
        * [Image Classification](#Image-Classification)
        * [Numeric Classification](#Numeric-Classification)
        * [Numeric Regression](#Numeric-Regression)


<a name="Examples"></a>
## Examples of StackML models
<img src="https://d3l5ba5b56ptip.cloudfront.net/media/image-classify-output.jpg" width="500">
> Image Classification

<img src="https://d3l5ba5b56ptip.cloudfront.net/media/yolo-output.jpeg" width="500">
> YOLO

<img src="https://d3l5ba5b56ptip.cloudfront.net/media/posenet-output.png" width="500">
> PoseNet

<img src="https://d3l5ba5b56ptip.cloudfront.net/media/bodypix-output.jpg" width="500">
> BodyPix

<img src="https://d3l5ba5b56ptip.cloudfront.net/media/face-detection.jpeg" width="500">
> Face Detection

<img src="https://d3l5ba5b56ptip.cloudfront.net/media/face-landmark-output.png" width="500">
> Face Landmark Detection

<img src="https://d3l5ba5b56ptip.cloudfront.net/media/face-expression-output.png" width="500">
> Face Expression Recognition

<a name="Getting-Started"/>
## Getting Started

#### Access key
Get your access key [here](https://stackml.com/docs/getting-started/)

#### Setup
Add the following script tag to your main HTML file to include StackML library.

```html
<script src="https://stackml.com/library-js/stackml.min.js">
```

<a name="How-To-Use"/>
## How to use
Below is a complete example to use Image classification model on an image using StackML library.

```html
<html>
<head>
    <head>
        <title>Getting Started with StackML</title>
        <script src="https://stackml.com/library-js/stackml.min.js"></script>
    </head>

<body>
<h1>An example to use Image classification model with StackML library</h1>
<p>The model recognized the image as <span id="className">...</span> with a confidence of <span id="probability">!!!</span></p>
<img src="https://s3-us-west-2.amazonaws.com/stackml/docs/images/red_fox.jpeg" crossorigin="anonymous" id="red_fox" width="500">

<script>
    callStackML();

    async function callStackML() {
        //provide the access key
        await stackml.init({'accessKeyId': '<YOUR ACCESS KEY>'});

        //load the model
        const model = await stackml.imageClassifier('MobileNet', callbackLoad);

        // make prediction with the image
        model.predict(document.getElementById('red_fox'), callbackPredict);

        // callback on model load
        function callbackLoad() {
            console.log('model loaded!');
        }

        // callback after prediction
        function callbackPredict(err, results) {
            //display the results
            document.getElementById('className').innerText =
                results['outputs'][0].className;
            document.getElementById('probability').innerText =
                results['outputs'][0].probability.toFixed(4);
        }
    }
</script>
</body>
</html>
```

<a name="Pre-trained-models"/>
## Pre-trained models
Pre-trained models are machine learning models which are already trained on a large number of dataset & are ready to use.

<a name="Image-Classification"/>
## Image Classification
Image classification is the simplest & widely used deep learning model. The model can take an image & assign one or more classes from a fixed set of categories (it's trained on). Image classification provided (MobileNet) is pre-trained on 15 million images spreaded into 1000 categories like airplane, dog etc.

Self driving cars are a great example to understand where image classification is used in real world.

**Loading the model**
```javascript
const classifier = await stackml.imageClassifier(modelType, callback?);
```
Parameters   | Description 
------------- | -------------
**modelType**   | Type of model to use for image classification. Available model: 'MobileNet'
**callback**   | Optional. Callback function gets called after the model load is complete.


**Predicting**
```javascript
classifier.predict(input, callback?, options?);
```
Parameters   | Description 
------------- | -------------
**input**   | HTML image or video element.
**callback**   | Optional. Callback function gets called after the model prediction is complete.
**options** | Optional. Additional option to change model performance & accuracy. Refer below table. Example - {numberOfClasses:3}


<a name="YOLO"/>
## YOLO
You only look once (YOLO) is an object detection model which helps to identify the location of an object in the image. It will output the coordinates of the location of an object with respect to the image.

It is widely used in computer vision task such as face detection, face recognition, video object co-segmentation.

**Loading the model**
```javascript
const yolo = await stackml.YOLO(callback?);
```
**Detecting objects**
```javascript
yolo.detect(input, callback?);
```
**Draw optput**
```javascript
yolo.draw(canvas, input, results);
```

<a name="PoseNet"/>
## PoseNet
PoseNet is a pose estimation model which detects human postures in images and video, so that one could determine, for example, where someone’s elbow shows up in an image. It can be used to estimate either a single pose or multiple poses.

PoseNet detects 17 key points to determine human body posture like nose, left-eye, left-ear, left shoulder... etc.

**Loading the model**
```javascript
const posenet = await stackml.poseNet(callback?);
```
**Detecting human poses**
```javascript
posenet.detect(input, callback?);
```
**Draw optput**
```javascript
posenet.draw(canvas, input, results);
```

<a name="BodyPix"/>
## BodyPix
BodyPix is a pre-trained model to perform Person Segmentation on an image or video. In other words, BodyPix can classify the pixels of an image into two categories: 1) pixels that represent a person and 2) pixels that represent background.

**Loading the model**
```javascript
const model = await stackml.bodypix(callback?);
```
**Segmenting human body from background**
```javascript
model.detect(input, callback?);
```
**Draw optput**
```javascript
model.draw(canvas, input, results);
```

<a name="Face-Detection"/>
## Face Detection
Face Detection model is used to detect human face in an image or video. It can be regarded as a specific case of object-class detection. Face Detection will output coordinates of box locating the human face.

**Loading the model**
```javascript
const model = await stackml.faceDetection(callback?);
```
**Detecting human faces**
```javascript
model.detect(input, callback?);
```
**Draw optput**
```javascript
model.draw(canvas, input, results);
```

<a name="Face-Landmark-Detection"/>
## Face Landmark Detection 
Face Landmark Detection also known as Face Feature Detection model detects 68 facial keypoints in an image.

**Loading the model**
```javascript
const model = await stackml.faceLandmark(callback?);
```
**Detecting human face key points**
```javascript
model.detect(input, callback?);
```
**Draw optput**
```javascript
model.draw(canvas, input, results);
```

<a name="Face-Expression-Recognition"/>
## Face Expression Recognition
Face Expression Recognition model detects human face & predicts whether the person is sad happy angry & so on. It will out the prediction in seven categories of expressions ie; neutral, happy, sad, angry, fearful, disgusted & surprised. It will also output coordinates of box locating the human face.

**Loading the model**
```javascript
const model = await stackml.faceExpression(callback?);
```
**Detecting human faces**
```javascript
model.detect(input, callback?);
```
**Draw optput**
```javascript
model.draw(canvas, input, results);
```

<a name="User-trained-models"/>
## User trained models
After you train a machine learning model on your own dataset using StackML platform, use following code to run prediction on it.

<a name="Image-Classification"/>
## Image Classification
Image classification is the simplest & widely used deep learning model. The model can take an image & assign one or more classes from a fixed set of categories (it's trained on). Image classifier provided (MobileNet) is pre-trained on 15 million images spreaded into 1000 categories like airplane, dog etc.

Self driving cars are a great example to understand where image classification is used in real world.

**Example**
```javascript
const modelPath = 'https://foo.bar/your_model.json';
await stackml.init({'accessKeyId': <YOUR ACCESS KEY>});
// load user image classifier model using Mobilenet
const model = await stackml.imageClassifier('MobileNet', callbackLoad, modelPath);

// make prediction with the image
model.predict(document.getElementById('image'), callbackPredict);

// callback on model load
function callbackLoad() {
    console.log('model loaded!');
}

// callback after prediction
function callbackPredict(err, results) {
    console.log(results);
}
```

**Loading the model**
```javascript
const model = await stackml.imageClassifier(modelType, callback?, modelPath);
```
**Predicting**
```javascript
classifier.predict(input, callback?, options?);
```

<a name="Numeric-Classification"/>
## Numeric Classification
Numeric classification is the simplest & widely used deep learning model. The model takes a csv & assign one or more classes from a fixed set of categories (it's trained on).

You can train numeric classification model on your own dataset.

**Loading the model**
```javascript
const model = await stackml.numericClassifier(modelPath, callback?);
```
**Predicting**
```javascript
classifier.predict(input, callback?, options?);
```

<a name="Numeric-Regression"/>
## Numeric Regression
Numeric regression is the simplest & widely used deep learning model. The model takes an input csv & calculates an output for each row.

**Loading the model**
```javascript
const model = await stackml.numericRegression(modelPath, callback?);
```
**Predicting**
```javascript
classifier.predict(input, callback?, options?);
```

### Terms of use
Read it [here](https://stackml.com/terms)

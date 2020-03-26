## 任务说明
- 搭建模型进行人脸识别；
- 最终模型可以与摄像头调用，并进表情识别。

## 数据源
[数据地址](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## 数据说明
- 数据中为48x48像素人脸灰度图。数据中人脸在居中位置，并且面积大致相等。需要做的就是将人脸表情进行分类包含七个类：(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
- train.csv 包含两列，第一列是表情，第二列是像素。表情列中使用数字0-6标注不同类。像素列中包含了每个图形的像素的字符串，存储方式为列优先存储。

## 代码
- data_load, 数据读写
- emotion.ipynb VGG模型
- emotion_all 完整实现


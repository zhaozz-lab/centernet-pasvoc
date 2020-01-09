# one-stage-object-detection
one stage object detection include yolov1 yolov2 yolov3 object as point
reference https://github.com/xingyizhou/CenterNet


## the code need to complile the DCNv2，this code is compliled in windows 10,VS2015 python3.6
## the file directory is:


## detect.py test the original model(train in coco dataset)
* testvoc.py test the result of train using the PASVOC dataset
## data preprocess
* download PASVOC dataset in the data dir
move the voc_label.py to the data dir
```
python voc_label.py
```
could generate the *.txt like   
   2007_test.txt  
   2007_train.txt  
   2007_val.txt  
   2012_train.txt  
   2012_val.txt 
```sh
cat 2007_train.txt 2012_train.txt 2012_val.txt 2007_val.txt>train.txt
```

```
cat 2007_test.txt >val.txt 
```

## train
train code is in the experiments
```sh
sh pasvoc_384_origin.sh
```
using the resnet18dcn the mAP is 0.69
using the resnet50dcn the mAP is 0.73
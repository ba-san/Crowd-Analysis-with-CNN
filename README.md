# Crowd-Analysis-with-CNN

For its network, I used WideResNet from [here](https://github.com/nabenabe0928/wide-resnet-pytorch).  

## Guidance
For detailed info, please check each directory.  

### population estimation
<img src="https://user-images.githubusercontent.com/44015510/64599784-fa84d100-d3f4-11e9-9f80-c33d37717a08.JPG" width="300"> 

**classification&regression** - estimate just number of people in cropped img by classification approach or regression approach  

**classification_test&regression_test** - can test trained network with different dataset you used for training.    

### density map
<img src="https://user-images.githubusercontent.com/44015510/64599799-007ab200-d3f5-11e9-821c-0a3b25c4da83.JPG" width="300"> 

**frame** - create density map  

**cell** - this is for comparison with frame.    

**video** - create density map video.  

### location estimation
<img src="https://user-images.githubusercontent.com/44015510/64599810-05d7fc80-d3f5-11e9-84e5-ef6e9fbce0f4.JPG" width="300"> 

**location** - with output from number of people estimating network (i.e. **classification&regression**), you can estimate exact coordinates of people in those imgs.  

## Requirements 
・python3.7  
・pytorch   
    `conda install pytorch torchvision cudatoolkit=x.x -c pytorch` 
    
## Instructions
First of all, you need to prepare crowd dataset. You can make it with my annotating script named [Count-Annotator2](https://github.com/ba-san/Count-Annotator2).  
Leave your dataset in 'dataset' directory.  

# Crowd-Analysis-with-CNN

For its network, I used WideResNet from [here](https://github.com/nabenabe0928/wide-resnet-pytorch).  

## Guidance
For detailed info, please check each directory.  

**classification&regression** - estimate just number of people in cropped img by classification approach or regression approach  

**classification_test&regression_test** - can test trained network with different dataset you used for training.    

**location** - with output from number of people estimating network (i.e. **classification&regression**), you can estimate exact coordinates of people in those imgs.  

**frame** - create density map  

**cell** - this is for comparison with frame.    

**video** - create density map video.  

## Requirements 
・python3.7  
・pytorch   
    `conda install pytorch torchvision cudatoolkit=x.x -c pytorch` 
    
## Instructions
First of all, you need to prepare crowd dataset. You can make it with my annotating script named [Count-Annotator2](https://github.com/ba-san/Count-Annotator2).  
Leave your dataset in 'dataset' directory.  

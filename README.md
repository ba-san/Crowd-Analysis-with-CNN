# Crowd-Analysis-with-CNN

For its network, I borrowed WideResNet from [here](https://github.com/nabenabe0928/wide-resnet-pytorch).  

## guidance
**classification&regression** - estimate just num of ppl in cropped img by classification approach or regression approach  
**classification-test&regression-test** - can test trained network with different dataset you used for training.    
**frame** - done  
**cell** - done  
**location** - with output from num of ppl estimating network (i.e. **classification&regression**), you can estimate exact coordinates of ppl in those imgs.  

## requirements 
・python3.7 
・pytorch 
    `conda install pytorch torchvision cudatoolkit=x.x -c pytorch` 

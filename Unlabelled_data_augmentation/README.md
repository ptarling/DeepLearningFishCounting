### Generating unlabelled image pairs for self-supervised task

The only manual pre-processing step required for generating unlabelled images is drawing an image bounding box to mark the geographical area to be sampled. Note this only has to be done once / video (or once / every ~1,000 images). I.e. to extract 50,000 pairs, from 50,000 different original images, ~50 bounding boxes will need to drawn - estimated user time = 25 minutes. 

Below are detailed instructions of how to do this with VGG Image Annotator (VIA) tool (https://www.robots.ox.ac.uk/~vgg/software/via/). The exported JSON file can be directly used with our script <a href="https://github.com/ptarling/DeepLearningFishes/blob/main/unlabelled_data_augmentation/unlabelled_data_augmentation.py">unlabelled_data_augmentation.py</a> for augmenting pairs of images to train the self-supervised task of our network. A different approach can be taken with minor modifications to our code. 

#### Steps to produce image bounding box coordinates with required JSON format:

1. Select video files for data generation and convert to image
2. Create a new project in VIA and upload ("Add file") 1 image / video
3. Draw a rectangular bounding box to mark the geographical sample region. (In our study, this range was 8m to 17m, an area of 4x8.5m<sup>2</sup>, - the base of the rectangular spanned the width at 8m.)
4. Export annotations as json file
5. Load .json file into <a href="https://github.com/ptarling/DeepLearningFishes/blob/main/unlabelled_data_augmentation/unlabelled_data_augmentation.py">unlabelled_data_augmentation.py</a> and run

#### Variables

1. Load [YOUR].json file with bounding box coordinates
2. std_width: default = 320. Choose pixel width to resize all images to. 320 was average pixel width of images sampled in labelled training set with geographical range 8m to 17m 
3. std_height: default = 576. Choose pixel height to resize all images to. NB the script crops the image to this height from resized images to ensure consistent geographical area between images (mitigating descrepencies in user drawing bounding box). 576 was derived from the smallest pixel height:width ratio of images sampled in labelled training set with geographical range 8m to 17m - the bounding box at a minimum captured this area, ensuring all mullet would be labelled in this region.  

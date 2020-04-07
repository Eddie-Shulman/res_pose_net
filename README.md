#Final Project - Intro To Computer Vision

## The problem - Face pose estimation
The problem was to estimate 3D face pose, i.e. euler angles, using computer vision\ machine learning\ deep learning 
techniques.
 We were required to:
 1. Create your own dataset of image/pose
 2. Develop a model, or several models and pick the best
 3. Analyze and document the work 

## Solution outline
There are 2 possible solutions to the face pose problem:
1. Find the facial landmarks and calculate the pose using pnp
2. Find the pose directly from the image

The first option is quite straight forward but then one might ask, how may landmarks should I use? Which ones? Does skin color, age or race influence those landmarks and how?

On the other hand, the 2nd option provides the freedom of selecting those landmarks, or any facial feature to that matter, to the learning system. Given enough data, those are generally out-perform the human engineered features. Thus I decided to peruse this option as my solution.

I chose to tackle this project using deep learning, mainly to learn and gain some experience. Plus, I thought I would have a large enough dataset to train the NN. I chose Terraform and Keras as my tech stack due to their simplicity.

My first step was creating a dataset large enough so it can be used to train a Neural Network(NN).

Then I chose a training method. I tried 2 simple custom NNs which I came up with and a 3rd NN based on RESNET.

Finally, I used the generated datasets as my training set and the given 2nd validation set as my validation set.

## Extending existing, annotated, datasets
As starting point, I used the 300w_3d HELEN and LFPW datasets and the AFLW2000 dataset. Initially they consist of 5,364 images.

My first plan was to use the given face model to generate augmentations of the images along with all the 3 rotation angles. I relied on the technique (and code) of FacePoseNet project. However, many of the images were a bit distorted and I couldn’t get good results with them thus I abandoned this way of augmentations.

I also considered to use scale as augmentation but since I was scaling images in pre-processing before training, the scale effect would have been minimal, thus I dropped this idea.

Finally, for each image in those datasets I have applied the following 2D transformations:
1. Rotation
2. Flip

I used several groups of rotation transformations with different angles (3 angles in each group) and applied only one group to an image. This way I was able to apply large variety of angles without using a single image too much.

In addition, I chose NOT to use the given pose, instead I used the image landmarks and the given face model to calculate the pose:
1. Calculate the projection matrix using cv2.solvePnP, using the transformed landmarks along with the given 3D face model (3D landmarks and the camera model).
2. Use the projection matrix to extract the RT matrix:
𝑀_𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥=𝐴_𝐶𝑎𝑚𝑒𝑟𝑎𝑀𝑜𝑑𝑒𝑙∙𝑅𝑇_𝑇𝑟𝑎𝑛𝑠𝑓𝑜𝑟𝑚𝑎𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥⇒𝐴_𝐶𝑎𝑚𝑒𝑟𝑎𝑀𝑜𝑑𝑒𝑙−1∙𝑀_𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥=𝑅𝑇_𝑇𝑟𝑎𝑛𝑠𝑓𝑜𝑟𝑚𝑎𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥 <br/>
Since 𝐴_𝐶𝑎𝑚𝑒𝑟𝑎𝑀𝑜𝑑𝑒𝑙 is given in the 3D face model, and the 𝑀_𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥 was calculated, I was able to calculate 𝑅𝑇_𝑇𝑟𝑎𝑛𝑠𝑓𝑜𝑟𝑚𝑎𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥 .
Extracting the translation vector is straight forward – it’s the last column in the 𝑅𝑇_𝑇𝑟𝑎𝑛𝑠𝑓𝑜𝑟𝑚𝑎𝑡𝑖𝑜𝑛𝑀𝑎𝑡𝑟𝑖𝑥.<br/>
Extracting the rotation angles was done using cv2.Rodrigues.

I chose this technique since different face and camera models will affect the results and generate errors. Since our results will be tested against the given face and camera model, this technique will minimize such errors.

Finally, I had 32,184 images which I hopped will be enough.

## Building a training process based on the original NN
As to the training method, my initial plan was to use the given pre trained Neural Network(NN), FacePoseNet, for transfer learning. However, after building a training pipe line I couldn’t make the NN to converge. After several days of tinkering with optimizers, loss functions and learning rate decay I gave up and turned to RESNET.

I hoped to use RESNETs inner layers to generate feature vectors. Even though it looks very simple, I came across several difficulties:
1. The images must go through RESNETs pre-processing which normalizes the data.
2. Even though it is very easy to remove RESNETS head, it’s inner layers are still optimized for classification of ImageNet. It took a while to figure which layers should be frozen, during training, and which shouldn’t
3. It is a large NN and I was skeptical whether it will learn and generalize or just over fit.

After some research, I figured which layers I should freeze (make non trainable) and which layers I should add as a new head:
1. BatchNormalization with ReLu activation
2. Droput with 0.3 rate
3. Dense with 1024 units and ReLu activation
4. Flatten
5. Droput with 0.3 rate
6. Dense with 3 units, no activation – my output layer

The dropout layers were supposed to handle the overfitting effect which I expected to happen due to low amount of images.
I used a similar head in my custom NN.

Training C_RESNET (my customized RESNET) on a small batch of training data (5K images) didn’t yield the promised results. Validation accuracy was at ~70% with an average of 25deg error. This could be due to the small training data but all could be a problem with this solution. Thus I decided to build CNN based custom NN as a validation point:
1. C1_NET – a simple NN which consists of Conv2D layers with MaxPooling2D between then. This small and simple NN outperformed C_RESNET with an accuracy of ~85% and an average of 10deg error.
2. C2_NET – a more complex NN which consists of more layers of Conv2D with BatchNormalization (with ReLu activation) between them. This NN outperformed C_RESNET as well with 81% accuracy and 12deg error

All the NNs were trained:
1. on image and pose (rx, ry, rz) as a label. I did try, at first, to train all the 6 pose parameters, but didn’t get good results thus returned to train only the ROTVEC.
2. With Adam optimizer
3. MSE loss function. I did try to use a custom loss function which reflected the angle error but it was too slow and sometimes the results was inf when the rotation matrixes were very close (np.trace(rot_mat1.T @ rot_mat2)~1.0000000001).

The Adam optimizer was selected to its ability to auto manage the decay rate for variables within the NN.

I did try to use SGD (Stochastic gradient descent) with momentum optimizer but it didn’t perform as well as Adam. May be adding decay on the learning rate would have improved the performance but I didn’t have the time to check that.

Another aspect of the training was preprocessing the serving the data.
In the preprocess phase I:
1. Got the face bbox (bounding box) from the image using dlib. As a contingency plan, I’ve added OpenCVs face detector DNN.
2. Crop the face according to the bbox and resize the image to 224, 224 (input for RESNET). First I made the mistake of distorting the face due to difference in aspect ratio, but that was easily sorted out, and the extra space of the image was filled with 0 pixels.
3. Run the Keras pre-process for RESNET.

In the serving data phase:
1. I have split the data into batches of 32 images
2. The batches were selected randomly
3. After each epoch, the data was shuffled and the batches updated

This was done in order to achieve maximum randomization of the learning process.

## Analyze results
To analyze the performance of each NN and to fine tune the hyper parameters I did several test training sessions on a fraction of my dataset, 5K images, and validated against the given validation set 2.

The metrics I was measuring were:
1. MSE loss
2. Accuracy
3. Custom loss
4. Validation loss
5. Validation accuracy
6. Validation custom loss

The custom loss is a calculation of the difference between 2 rotation matrix calculated from the labeled ROTVEC (rotation vector – rx, ry, rz) and the predicted ROTVEC. Thus as close it is to 0 the less error there is between the label and the prediction.

I used Tensorboard to monitor the training and validate the results.
An example of the test training session with C_RESNET:<br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/c_resnet_50_part.jpg "Resnet, 50 epoch, partial training")

From the data we can see a constant improvement on all the metrics. However, the validation accuracy values are a bit off. This is due to the way measuring accuracy is defined. Thus I did monitor the trend, to verify the training process is progressing, but didn’t give any attention to the actual values.

Another example from a training session of the C2_NET:<br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/c2_net_50_part.jpg "C2_NET, 50 epoch, partial training")

The difference in performance is evident, but to compare:<br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/val_custom_acc_diff.jpg "Resnet vs C2_NET, 50 epoch, partial training")

And the finaly results, trained on the full dataset:<br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/c2_net_50.jpg "Resnet, 50 epoch")

I did continue to train the net for additional 30 epochs, descressing 5 times the learning rate. Sure I got way better results, ~5deg avg. error, however, since training accuracy improved only by 5% and validation accuracy improved by nearly 20% I figured the model started to overfit.<br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/c2_net_50_vs_80.jpg "C2_NET, 50 epoch vs 80 epoch")

C_RESNET, epoch 0 to 80 <br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/c2_net_80.jpg "Resnet, 80 epoch")

On the validation set, C2_NET outperformed C_RESNET, even though it was trained only 50 epochs while C_RESNET was trained 80 epochs :<br/>
![alt text](https://raw.githubusercontent.com/Eddie-Shulman/res_pose_net/master/results/c_resnet_vs_c2_net.jpg "Resnet, 80 epoch, vs C2_NET, 50 epoch")

I did continue to train the net for additional 30 epochs as well, descressing 5 times the learning rate, as I did with C2_NET. Once again I got better results, ~15deg avg. error in appose to ~17deg error, however, since training accuracy improved only by 2% and validation accuracy improved by 6% I figured the model started to overfit aswell.

Finally, I tried to estimate the performance on the given test set:
1. I used dlib to extract face bbox and face landmarks
2. I used the given face model and the calculated landmarks to get the rotation and translation (using pnp), thus the pose.

The results I got were not assuring, to say the least:

|           | C2_NET, 50 epochs | C_RESNET, 80 epochs | C2_NET, 2nd training | C_RESNET, 2nd training |
|:---------:|:-----------------:|:-------------------:|:--------------------:|:----------------------:|
| deg error | ~50               | ~41                 | ~46                  | ~47


Those results were surprising as I thought my C2_NET will perform better, but I gues it was overfitting during training.

The big difference between the predicted and the “labled” values can be explained by the fact that the The landmarks calculated by dlib were not as accurate as the manaul labels of the training set. Hopefully this is the case.

And so the attached results, results.csv, were generated by C_RESNET (1st training only).           

## Resources
Datasets:
1. 300W_3D – HELEN and LFPW datasets, downloaded from [here](https://drive.google.com/file/d/0B7OEHD3T4eCkRFRPSXdFWEhRdlE/view)
2. AFLW2000 dataset, can be downloaded from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip) 

Related work:
1. [FacePoseNet: Making a Case for Landmark-Free Face Alignment](https://arxiv.org/abs/1708.07517)
2. [Dlib: Face detection and recognition](http://dlib.net)

## Disclaimer & License
The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use.

This project is licensed under [GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)
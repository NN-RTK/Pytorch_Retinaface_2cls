# retinaface_2cls

# Aim: 
Repository to train, evaluate, and run inference using a modified retinaface model.

# Background: 
The original retinaface is trained to detect faces and 5 landmark points. It performs very good for small face objects.
To leverage the small object detection performance, and extend it for object detection task, retinaface is modified to detect two objects (faces and humans). 
Furthermore, we do not need landmarks for general object detection. 
So, it is proposed to modify retinaface such that it outputs bounding boxes for two classes and it does not output any landmarks. 

# Modification in model:
1. Drop the landmark calculation head, and landmark regression loss calculation
2. Change the output dimensions of the classification head to output 2 classes + 1 bg
3. Calculate and modify priors to cover the ground truth of "person" class and make required subsequent changes in the network dimensions
4. Modify loss

# Dataset creation:
1. Create 'person' class ground truth using inference from other models.
   -> Observation : The model performs poorly on person class and learn the features.
   -> Conclusion : Widerface dataset contains images with crowds and overlapping person images. Also get some wrong inferences.
2. Download "Human face" and "Person" classes from open images dataset (OID) -> Generate ground truth in same format as widerface
3. Analyse ground truth to calculate priors/anchor boxes
4. Modify MAP calculation script to accept two classes and evaluate the models
   -> Observation : Performance drop  for face
   -> Conclusion : Missed and incorrect ground truth , Open Images Dataset also contains some entries with 'Man, Woman, Boy, Girl' labels.
5. Add images with 'Man, Woman, Boy, Girl' labels under "Person"
   -> Observed missed "Face" annotations in the added entries
6. Add "Face" annotations using inference with retina face
    -> Observed out-of-image-boundary detections for objects at the image boundary. (negative and >1)
7. Cleaned ground truth from out-of-boundary issues and repeted detections by overlap calculation.  

# Run scripts and their parameters
1. data_prep
   Crate a symbolic link for an "images" folder in train and val folders with the training and validation images in the dataset repository
   ```
   ln -s <path/to/train/or/validation/images> "./data/wider_OIpersonface1103/train/images"
   ```
3. train
   ```
   CUDA_VISIBLE_DEVICES=0 python3 train_changeclass.py --network mobile0.25 --resume_net "/home/ruchi/work/models/Pytorch_Retinaface-master/weights/mobilenet0.25_2cls6anchors_130324_epoch_150.pth" --resume_epoch 150
   ```
4. detect
5. test
   * script on line 197 - 200 - Normalise for wider_openimagespersonface_evaluate 
   ```
   x = int(b[0]) / im_shape[0]
   y = int(b[1])/ im_shape[1]
   w = (int(b[2]) - int(b[0]) )/ im_shape[0]
   h = (int(b[3]) - int(b[1]) )/ im_shape[1]
   ```
   * script on line 197 - 200 - Do not normalise for widerface_evaluate 
   ```
   x = int(b[0]) #/ im_shape[0]
   y = int(b[1])#/ im_shape[1]
   w = (int(b[2]) - int(b[0]) )#/ im_shape[0]
   h = (int(b[3]) - int(b[1]) )#/ im_shape[1]
   ```
   * script on line 216 - 220
   ```
   if not os.path.exists("./results_wider_OIpersonface_150_1803/"):#_OIperson
      os.makedirs("./results_wider_OIpersonface_150_1803/") #_openimagesperson#_OIperson
   name = "./results_wider_OIpersonface_150_1803/" + str(i) + ".jpg"   
   cv2.imwrite(name, img_raw)
   ```
   * parameters
   ```
   --save_folder './widerface_evaluate/results_widerface_150_1803txt/' 
   OR --save_folder './wider_openimagespersonface_evaluate/results_wider_OIpersonface_150_1803txt/' 

   --dataset_folder './data/widerface/val/images' 
   OR  --dataset_folder "./data/wider_OIpersonface1103/val/images", type=str, help='dataset path') #
   
   ```

   python3 test_widerface_personface.py --trained_model weights/mobilenet0.25_2cls6anchors_130324_epoch_150.pth --network mobile0.25

6. evaluation
   ```
   cd widerface_evaluate OR cd wider_openimagespersonface_evaluate
   python3 evaluation.py
   ```
   
# Evaluation:
## Evaluate 'face' class on widerface dataset
1. Retinaface original model
* Easy   Val AP: 0.907077
* Medium Val AP: 0.8816508
* Hard   Val AP: 0.738287

3. Retinaface_2cls trained on initial dataset 2k interations
* Easy   Val AP: 0.6754746
* Medium Val AP: 0.4721576
* Hard   Val AP: 0.20005402

5. Retinaface_2cls trained on clean dataset 50 iterations @ IoU 0.5
   1. observation : face annotation with and without hair changes the IoU, hence also evaluated using IoU0.3 in next
   2. observation : takes much longer because of more images
* Easy   Val AP: 0.8339701978904259
* Medium Val AP: 0.74449873143092
* Hard   Val AP: 0.3896353350813585

6. Retinaface_2cls trained on clean dataset 50 iterations @ IoU 0.3
* Easy   Val AP: 0.85099
* Medium Val AP: 0.7592925
* Hard   Val AP: 0.400277883

8. Retinaface_2cls trained on clean dataset 150 iterations @ IoU 0.5
   1. note: lowered the learning rate to 0.0001 from 0.001 because the loss stabilised
   2. observation : the performance seems to not have improved only by a couple percentages 
* Easy   Val AP: 0.8495088187
* Medium Val AP:0.77570073166
* Hard   Val AP: 0.418763877

9. Retinaface_2cls trained on clean dataset 250 iterations @ IoU 0.3
* Easy   Val AP:0.867159132324704
* Medium Val AP: 0.7913999214072
* Hard   Val AP: 0.4310744194


## Evaluate 'face' and 'human' classes on open images dataset
1. Retinaface_2cls trained on on initial dataset 2k interations of original OID
   * Face: 0.3831289624539691
   * Person: 0.4756916624444069

2. Retinaface_2cls trained on cleaned OID : 150 iterations @ IoU 0.5, LR = 0.0001
   * Face: 0.45883516187151224
   * Person: 0.4229204320016947
3. Retinaface_2cls trained on cleaned OID : 150 iterations @ IoU 0.3, LR = 0.0001
   * Face: 0.5365842974876509
   * Person: 0.47603287699017527

   

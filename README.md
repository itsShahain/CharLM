A character level language model built in varying complexities from the simplest onwards:

Bigram Model
MLP (with manual backprop - i.e no loss.backward() 💪🏼)
RNNs, LSTM + GRU

Note: this repo is still under development


# Training Pipeline of Our Model

The training pipeline of our model is as follows:

## a) Class-Agnostic Detection Transformer Training

First we train the class-agnostic Detection Transformer on CAPTCHA images with bounding boxes drawn manually by us. After manual labelling and data augmentations, we had about ~1,000 bounding-box drawn images. To train, please run (on the Tembusu cluster):

```bash
srun --gpus=a100-40:1 python main.py --dataset_file letter --data_path ../dataset/ --output_dir output --resume weights/detr-r50-e632da11.pth
```

Final training metrics (please refer to `slurm-578307-DETR_TRAIN_LOG.out`):

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.741
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.957
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.902
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.744
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.143
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.801
```

## b) Segment & Extract Characters

Then this trained model is used to segment and extract individual characters from the cleaned CAPTCHA images (we manually looked through the train dataset and removed all corrupted CAPTCHAS that either contained wrong labels, Greek letters, non-standard symbols etc). After segmentation using the class-agnostic Detection Transformer, we obtained about ~17,000 individual letter images (see `sample_segmentations` for examples of segmented letters).

To replicate our extractions, please run:

```bash
srun --gpus=a100-40 python extract.py --data_path ../dataset/cleaned_images --resume old_output/checkpoint.pth --output_dir today
```

## c) Train CNN Experts

Then the individual CNN's that comprise the simulated Baye's optimal ensemble are trained on these segmented individual letters.

```bash
# Example training command for CNN experts
python train_cnn.py --data_dir today/ --output_dir cnn_weights/
```

## d) Compute each CNN Expert's Prior Weights

We create prior weights for each expert using the validation accuracy on the custom ~17,000 letter dataset. We give each expert a weight of:

```
weight_i = accuracy_i / (sum of all validation accuracies of all networks)
```

## e) Final Pipeline Evaluation

We finally evaluate the entire pipeline by running:

```bash
srun --gpus=a100-40:1 python -u finale.py --data_path ../dataset/amir_test/test --resume old_output/checkpoint.pth
```

The metrics achieved are (you can check the slurm files `slurm-597563-PIPELINE_PERFORMANCE_TEST_1.out` and `slurm-602124-PIPELINE_PERFORMANCE_TEST_2.out`):

```
Successfully processed images: 2000
Avg. Time per processed image: 0.017s
Correctly predicted : 9547
Total number of characters : 11733
0.8136878888604789
Total number of captchas : 2000
Correctly predicted : 881
0.4405
Precision: 0.8137
Recall:    0.7949
F1 Score:  0.8042


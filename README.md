# :ocean: MSResNet-via-SSL
## :page_facing_up: Paper
MSResNet: Multi-Scale Residual Network via Self-Supervised Learning for Water-Body Detection in Remote Sensing Imagery  
:exclamation: Our paper are available at [link](https://www.mdpi.com/2072-4292/13/16/3122#) (Open Access)

## :key: Abstract
Driven by the urgent demand for flood monitoring, water resource management and environmental protection, water-body detection in remote sensing imagery has attracted increasing research attention. Deep semantic segmentation networks (DSSNs) have gradually become the mainstream technology used for remote sensing image water-body detection, but two vital problems remain. One problem is that the traditional structure of DSSNs does not consider multiscale and multishape characteristics of water bodies. Another problem is that a large amount of unlabeled data is not fully utilized during the training process, but the unlabeled data often contain meaningful supervision information. In this paper, we propose a novel multiscale residual network (MSResNet) that uses self-supervised learning (SSL) for water-body detection. More specifically, our well-designed MSResNet distinguishes water bodies with different scales and shapes and helps retain the detailed boundaries of water bodies. In addition, the optimization of MSResNet with our SSL strategy can improve the stability and universality of the method, and the presented SSL approach can be flexibly extended to practical applications. Extensive experiments on two publicly open datasets, including the 2020 Gaofen Challenge water-body segmentation dataset and the GID dataset, demonstrate that our MSResNet can obviously outperform state-of-the-art deep learning backbones and that our SSL strategy can further improve the water-body detection performance.


## :star: Introduction & Graphical abstract
![GA](https://github.com/Jack-bo1220/MSResNet-via-SSL/blob/master/GA.tif)


## :large_blue_diamond: Training
- Download the code we released and then configure the python (Python 3.7) and deep learning environment (Pytorch 1.7.1).

    ~~~console
    git clone https://github.com/Jack-bo1220/MSResNet-via-SSL.git
    ~~~
- Prepare the training dataset, including images and the corresponding binary ground truth labels (distinguish between water and non water).
- Add the file path of training dataset images (**_dir_img_**) and labels (**_dir_mask_**) to the training script (**_train.py, line 17-18_**). In addition, if you need to use our proposed self-supervised learning method, you also need to add the file path of the unlabeled images to the **_./utils/dataset_2.py, line 60_**.
- The **_in_channels_** of our proposed MSResNet (**_./network/MSResNet.py, line 139_**) needs to be changed according to the number of bands of the training data. (default is 3)
- Run training script.
    ```
    python train.py -e $max_epoch -b $batch_size -l $learning_rate -f $pretrained_weights
    ```
- Add the file path of the validation dataset and inferencing results in the corresponding position in the validation script (**_predict_fwiou.py, line 107 117 163 166_**), so as to select the optimal model according to the performance of the validation dataset during the training process.


## :large_orange_diamond: Testing
- Download the [pretrained model](https://pan.baidu.com/s/1EPAWAF6gIc_FvWi8n_TOdA) (extraction code: 2020) we released or train own dataset to get the appropriate inference weight. （The pretrained model is trained by the 2020 Gaofen Challenge water-body segmentation dataset including RGB bands.）
- Run testing script.
    ```
    python predict.py -m $model_checkpoint -i $input_dir -o $output_dir
    ```
- Add the file path of inference results (**_path1_**) and test dataset labels (**_path2_**) in **_FWIoU2.py, line 54 56_**. Test segmentation accuracy. （optional）
    ```
    python FWIoU2.py
    ```


## :heavy_check_mark: Citation
If you find our source code helpful in your work, please cite the following papers:

[1] Dang, B.; Li, Y. MSResNet: Multiscale Residual Network via Self-Supervised Learning for Water-Body Detection in Remote Sensing Imagery. Remote Sens. 2021, 13, 3122. https://doi.org/10.3390/rs13163122  
[2] Li, Y., Shi, T., Zhang, Y., Chen, W., Wang, Z., & Li, H. (2021). Learning deep semantic segmentation network under multiple weakly-supervised constraints for cross-domain remote sensing image semantic segmentation. ISPRS Journal of Photogrammetry and Remote Sensing, 175, 20-33. https://doi.org/10.1016/j.isprsjprs.2021.02.009


## :smiley: Contact
If you have any questions about it, please feel free to let me know. (:email: email:bodang@whu.edu.cn)

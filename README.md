
Forked repo of the source code for the paper "Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation"

The paper can be found at [this link](https://ieeexplore.ieee.org/document/8964956).

## Overview
- [Dataset](#Dataset)
- [Pre-processing](#Pre-processing)
- [Architecture](#Architecture)
- [Training Process](#Training-Process)
- [Results](#Results)
- [Usage](#Usage)

### Dataset
The [BraTS](http://www.med.upenn.edu/sbia/brats2018.html) data set is used for training and evaluating the model. This dataset contains four modalities for each individual brain, namely, T1, T1c (post-contrast T1), T2, and Flair which were skull-stripped, resampled and coregistered. For more information, please refer to the main site.

### Pre-processing
For pre-processing the data, firstly, [N4ITK](https://ieeexplore.ieee.org/abstract/document/5445030) algorithm is adopted on each MRI modalities to correct the inhomogeneity of these images. Secondly, 1% of the top and bottom intensities is removed, and then each modality is normalized to zero mean and unit variance.


### Architecture
<br />

![image](https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/model.jpg)

<br />

The network is based on U-Net architecture with some modifications as follows:
- The minor modifications: adding Residual Units, strided convolution, PReLU activation and Batch Normalization layers to the original U-Net
- The attention mechanism: employing [Squeeze and Excitation Block](https://arxiv.org/abs/1709.01507) (SE) on concatenated multi-level features. This technique prevents confusion for the model by weighting each of the channels adaptively (please refer to [our paper](https://ieeexplore.ieee.org/document/8964956) for more information).

<br />

<p align="left"><img src="https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/attention.jpg" width="500" height="220"></p>

<br />

### Modifications

Few changes have been made to employ tensorpack module's argscope and pyfunctional module's chained sequence processing which helped in avoiding boilerplate code. 
Tensorpack module can be built from [my fork](https://github.com/gopalakrishna-r/tensorpack.git)

### Training Process
Since their proposed network is a 2D architecture, we need to extract 2D slices from 3D volumes of MRI images. To benefit from 3D contextual information of input images, we extract 2D slices from both Axial and Coronal views, and then train a network for each view separately. In the test time, we build the 3D output volume for each model by concatenating the 2D predicted maps. Finally, we fuse the two views by pixel-wise averaging.

<br />

<p align="left"><img src="https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/MultiView.jpg" width="600" height="220"></p>

<br />

### Results
The results are obtained from the [BraTS online evaluation platform](https://ipp.cbica.upenn.edu/) using the BraTS 2018 validation set.

<br />

<p align="center"><img src="https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/table.jpg" width="500" height="130"></p>

<br />

![image](https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/example.jpg)

<br />

### Dependencies
- [numpy 1.21.5](https://numpy.org/)
- [nibabe 3.2.2](https://nipy.org/nibabel/)
- [scipy 1.4.1](https://www.scipy.org/)
- [tables 3.7.0](https://www.pytables.org/)
- [Tensorflow 2.2.0](https://www.tensorflow.org/)
- [Keras 2.8.0](https://keras.io/)

### Usage
1- Download the BRATS 2019, 2018 or 2017 data by following the steps described in [BraTS](https://www.med.upenn.edu/cbica/brats2019/registration.html)

2- Perform N4ITK bias correction using [ANTs](https://github.com/ANTsX/ANTs), follow the steps in [this repo](https://github.com/ellisdg/3DUnetCNN) (this step is optional)

3- Set the path to all brain volumes in `config.py` (ex: `cfg['data_dir'] = './BRATS19/MICCAI_BraTS_2019_Data_Training/*/*/'`)

4- To read, preprocess and save all brain volumes into a single table file:
```
python prepare_data.py
```

5- To Run the training:
```
python train.py
```
The model can be trained from `axial`, `saggital` or `coronal` views (set `cfg['view']` in the `config.py`). Moreover, K-fold cross-validation can be used (set `cfg['k_fold']` in the `config.py`)


6- To predict and save label maps:
```
python predict.py
```
The predictions will be written in .nii.gz format and can be uploaded to [BraTS online evaluation platform](https://ipp.cbica.upenn.edu/).

### Citation

```
@inproceedings{noori2019attention,
  title={Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation},
  author={Noori, Mehrdad and Bahri, Ali and Mohammadi, Karim},
  booktitle={2019 9th International Conference on Computer and Knowledge Engineering (ICCKE)},
  pages={269--275},
  year={2019},
  organization={IEEE}
}
```

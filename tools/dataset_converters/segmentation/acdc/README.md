# Preparing Automated Cardiac Diagnosis Challenge

<!-- [DATASET] -->

```bibtex
@InProceedings{Agustsson_2017_CVPR_Workshops,
    author = {Agustsson, Eirikur and Timofte, Radu},
    title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {July},
    year = {2017}
}
```

[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) involves 100 patients, with the cavity of the right ventricle, the myocardium of the left ventricle and the cavity of the left ventricle to be segmented. Each case’s labels involve left ventricle (LV), right ventricle (RV) and myocardium (MYO).The database is made available to participants through two datasets from the dedicated online evaluation website after a personal registration: i) a training dataset of 100 patients along with the corresponding manual references based on the analysis of one clinical expert; ii) a testing dataset composed of 50 new patients, without manual annotations but with the patient information given above. The raw input images are provided through the Nifti format.
### Prepare dataset
To preprocess the ACDC data, you first need to download `training.zip` from https://acdc.creatis.insa-lyon.fr/#phase/5846c3ab6a3c7735e84b67f2
```
unzip training.zip
mkdir data/ACDCDataset
python tools/prepare_acdc.py training/
```
The dataset will be automatically automatically preprocessed. The file structure is as follows:

Note that we merge the original val dataset (image names from 0801 to 0900) to the original train dataset (image names from 0001 to 0800). The folder structure should look like:

```text
mmipt
├── mmipt
├── tools
├── configs
├── data
│   ├── ACDCDataset
│   │   ├── clean_data
│   │   │   ├── labelsTr
│   │   │   │   ├──patient001_frame13_0000.nii.gz
│   │   │   │   ├──patient002_frame13_0000.nii.gz
│   │   │   │   ├──patient003_frame13_0000.nii.gz
│   │   │   │   │──........
│   │   │   │   ├──patient015_frame13_0000.nii.gz
│   │   │   ├── imagesTr
│   │   │   │   ├──patient001_frame13_0000.nii.gz
│   │   │   │   ├──patient002_frame13_0000.nii.gz
│   │   │   │   ├──patient003_frame13_0000.nii.gz
│   │   │   │   │──........
│   │   │   │   ├──patient015_frame13_0000.nii.gz
│   │   ├── ACDCDataset_phase
│   │   │   ├── images
│   │   │   │   ├── patient030_frame12_0000.npy
│   │   │   │   └── ...
│   │   │   ├── labels
│   │   │   │   ├── patient030_frame12_0000.npy
│   │   │   │   └── ...
│   │   │   ├── train_list.txt
│   │   │   └── val_list.txt
```

## Crop sub-images

For faster IO, we recommend to crop the DIV2K images to sub-images. We provide such a script:

```shell
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

Then you can start the training program, such as the following command:
```
python train.py --config configs/acdc/nnformer_acdc_160_160_14_250k.yml --save_interval 250 --num_workers 4 --do_eval --log_iters 250 --sw_num 1 --is_save_data False --has_dataset_json False
```
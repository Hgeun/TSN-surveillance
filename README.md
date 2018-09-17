# TSN-Pytorch
# Using for surveillance system
*Now in experimental release, suggestions welcome*.

**Note**: always use `git clone --recursive https://github.com/yjxiong/tsn-pytorch` to clone original tsn-pytorch. 

## Training

Dataset : Trimmed UCF Crimes - https://github.com/henniekim/action_recognition_study/wiki/Temporal-annotation-for-UCF-Crimes-dataset


**Note**: How I set for each model is in table.  
  
For RGB models:

  | seg5 | seg10 | seg15 | seg20
:--: | :--: | :--: | :--: | :--:
dropout 0.4 | x | o | o | o
dropout 0.6 | o | o | o | o
dropout 0.8 | x | o | o | o

```bash
/datahdd/workdir/hyeongeun/anaconda3/envs/test2/bin/python3.6 \
/datahdd/workdir/hyeongeun/test/tsn-pytorch/main.py ucf101 RGB \
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_trainingset.txt \
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_testset.txt --arch BNInception \
--num_segments 10 --gd 20 --lr 0.0001 --lr_steps 30 60 --epochs 300 -b 32 -j 2 \
--dropout 0.6 --snapshot_pref re_RGB_seg10_drop06
```
 

For flow models:
  
  | seg5 | seg10 | seg15 | seg20
:--: | :--: | :--: | :--: | :--:
dropout 0.6 | o | o | o | o
  
```bash
/datahdd/workdir/hyeongeun/anaconda3/envs/test2/bin/python3.6 \
/datahdd/workdir/hyeongeun/test/tsn-pytorch/main.py ucf101 Flow \
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_trainingset.txt \
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_testset.txt --arch BNInception \
--num_segments 10 --gd 20 --lr 0.0001 --lr_steps 30 60 --epochs 300 -b 32 -j 2 \
--dropout 0.6 --snapshot_pref re_Flow_seg10_drop06
```

For Warp models:

  | seg10 | seg15 | seg20
:--: | :--: | :--: | :--:
dropout 0.6 | o | o | o

```bash
/datahdd/workdir/hyeongeun/anaconda3/envs/test2/bin/python3.6 \
/datahdd/workdir/hyeongeun/test/tsn-pytorch/main.py ucf101 Flow \ 
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_warp_trainingset.txt \ 
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_warp_testset.txt --arch BNInception \
--num_segments 10 --gd 20 --lr 0.0001 --lr_steps 30 60 --epochs 300 -b 16 -j 2 \
--dropout 0.6 --snapshot_pref re_Warp_seg10_drop06
```

## Testing

After training, there will checkpoints saved by pytorch, for example `re_RGB_seg10_drop06_checkpoint.pth`.
Also, there will best point saved, for example `re_RGB_seg10_drop06_best.pth`

Use the following command to test its performance in the standard TSN testing protocol:
  
for RGB models:

```bash
/datahdd/workdir/hyeongeun/anaconda3/envs/test2/bin/python3.6 \
/datahdd/workdir/hyeongeun/test/tsn-pytorch/test_models.py ucf101 RGB \
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_testset.txt \ 
/datahdd/Dataset/Trimmed_UCF_Crimes/TSN_result/re_RGB_seg10_epoch300/re_RGB_seg10_drop06_best.pth \
--test_segments 10 --arch BNInception --save_scores re_RGB_seg10_dp06
```

for Flow models:
 
```bash
/datahdd/workdir/hyeongeun/anaconda3/envs/test2/bin/python3.6 \
/datahdd/workdir/hyeongeun/test/tsn-pytorch/test_models.py ucf101 Flow \
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_testset.txt \ 
/datahdd/Dataset/Trimmed_UCF_Crimes/TSN_result/re_Flow_adam_seg10/re_flow_seg10_best.pth \
--test_segments 10 --arch BNInception --save_scores re_Flow_seg10_dp06
```
  
for Warp models:  

```bash
/datahdd/workdir/hyeongeun/anaconda3/envs/test2/bin/python3.6 \
/datahdd/workdir/hyeongeun/test/tsn-pytorch/test_models.py ucf101 Flow \ 
/datahdd/Dataset/Trimmed_UCF_Crimes/trim_shooting_re_warp_testset.txt \ 
/datahdd/Dataset/Trimmed_UCF_Crimes/TSN_result/re_Warp_seg10/re_Warp_seg10_drop06_best.pth \
--test_segments 10 --dropout 0.6 --arch BNInception --save_scores re_Warp_seg10_dp06
```

## Testing result
for RGB models:  

  |   | seg5 | seg10 | seg15 | seg20
:--: | :--: | :--: | :--: | :--: | :--:
dropout 0.4 | top1 |   | 41.71% | 36.57% | 40%
 .  | top3 |   | 59.43% | 57.14% | 61.14%
dropout 0.6 | top1 | 37.71% | 43.43% | 41.14% | 40%
 .  | top3 | 55.43% | 59.43% | 62.86% | 60.57%
dropout 0.8 | top1 |   | 40.57% | 38.29% | 36.57%
 .  | top3 |   | 61.71% | 59.43% | 57.71%



for Flow models:  

  |   | seg5 | seg10 | seg15 | seg20
:--: | :--: | :--: | :--: | :--: | :--:
dropout 0.6 | top1 | 42.29% | 39.43% | 35.43% | 38.86%
 .  | top3 | 63.43% | 60.57% | 59.43% | 61.14%



for Warp models:  

  |   | seg10 | seg15 | seg20
:--: | :--: | :--: | :--: | :--:
dropout 0.6 | top1 | 26.29% | 30.29% | 30.29%
 .  | top3 | 54.86% | 55.43% | 53.14%

## TSN-torch env pkg list
```bash
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
blas=1.0=mkl
ca-certificates=2018.03.07=0
certifi=2018.4.16=py36_0
cffi=1.11.5=py36h9745a5d_0
cuda80=1.0=0
cudatoolkit=8.0=3
cudnn=7.0.5=cuda8.0_0
freetype=2.8=hab7d2ae_1
intel-openmp=2018.0.0=8
jpeg=9b=h024ee3a_2
libedit=3.1.20170329=h6b74fdf_2
libffi=3.2.1=hd88cf55_4
libgcc-ng=7.2.0=hdf63c60_3
libgfortran-ng=7.2.0=hdf63c60_3
libpng=1.6.34=hb9fc6fc_0
libstdcxx-ng=7.2.0=hdf63c60_3
libtiff=4.0.9=he85c1e1_1
mkl=2018.0.2=1
mkl_fft=1.0.1=py36h3010b51_0
mkl_random=1.0.1=py36h629b387_0
ncurses=6.1=hf484d3e_0
numpy=1.14.3=py36hcd700cb_1
numpy-base=1.14.3=py36h9be14a7_1
olefile=0.45.1=py36_0
openssl=1.0.2o=h20670df_0
pillow=5.1.0=py36h3deb7b8_0
pip=10.0.1=py36_0
pycparser=2.18=py36hf9f622e_1
python=3.6.5=hc3d631a_2
pytorch=0.3.1=py36_cuda8.0.61_cudnn7.0.5_2
pyyaml=3.12=py36hafb9ca4_1
readline=7.0=ha6073c6_4
scikit-learn=0.19.1=py36h7aa7ec6_0
scipy=1.1.0=py36hfc37229_0
setuptools=39.1.0=py36_0
six=1.11.0=py36h372c433_1
sqlite=3.23.1=he433501_0
tk=8.6.7=hc745277_3
torchvision=0.2.0=py36_0
wheel=0.31.1=py36_0
xz=5.2.4=h14c3975_4
yaml=0.1.7=had09818_2
zlib=1.2.11=ha838bed_2
```

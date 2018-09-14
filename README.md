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

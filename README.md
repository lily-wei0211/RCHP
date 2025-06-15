# Few-shot 3D Point Cloud Semantic Segmentation via Relation Consistency guided Heterogeneous Prototypes [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10948324)]


## Overview

![framework](framework3d.png)



## Running 

**Installation and data preparation please follow [attMPTI](https://github.com/Na-Z/attMPTI).**



### Training

Pretrain the segmentor which includes feature extractor module on the available training set:

```bash
bash scripts/pretrain_segmentor.sh
```

Train our method under few-shot setting:

```bash
bash scripts/train_RCHP.sh
```

### Evaluation

Test our method under few-shot setting:

```bash
bash scripts/eval_RCHP.sh
```

Note that the above scripts are used for 2-way 1-shot on S3DIS (S^0). Please modify the corresponding hyperparameters to conduct experiments on other settings. 


## Quick Use

### Step 1: Add Testing Episode Files

Place the testing episode files for [S3DIS](#) into the `datasets` directory.

📁 Shared File: **S3DIS.zip**  
🔗 Baidu Drive: [https://pan.baidu.com/s/1APR_6PUyvhNbDHI8ugnMqQ?pwd=6aez](https://pan.baidu.com/s/1APR_6PUyvhNbDHI8ugnMqQ?pwd=6aez)  
🔑 Extraction Code: `6aez`

### Step 2: Add Our Checkpoints

Place the provided checkpoint and log files for [RCHP](#) into the [`log_s3dis`](https://github.com/lily-wei0211/RCHP/tree/main/log_s3dis) directory.

📁 Shared File: **RCHP.zip**  
🔗 Baidu Drive: [https://pan.baidu.com/s/10c8D8gERovIH8RPhAmsp-g?pwd=csfh](https://pan.baidu.com/s/10c8D8gERovIH8RPhAmsp-g?pwd=csfh)  
🔑 Extraction Code: `csfh`

### Step 3: Run Evaluation
Use the following command to evaluate the model:

## Checkpoints and Log Files
More checkpoints and log files will be released soon. Please stay tuned!

## Citation
Please cite our paper if it is helpful to your research:

    @ARTICLE{10948324,
      author={Wei, Lili and Lang, Congyan and Xu, Zheming and Liang, Liqian and Liu, Jun},
      journal={IEEE Transactions on Multimedia}, 
      title={Few-Shot 3D Point Cloud Segmentation via Relation Consistency-Guided Heterogeneous Prototypes}, 
      year={2025},
      volume={27},
      number={},
      pages={3158-3170},
      doi={10.1109/TMM.2025.3557699}
    }




## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch), [attMPTI](https://github.com/Na-Z/attMPTI), [QGPA](https://github.com/heshuting555/PAP-FZS3D), and [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP) for sharing their source code.

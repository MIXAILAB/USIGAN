
# USIGAN: Unbalanced Self-Information Feature Transport for Weakly Paired Image IHC Virtual Staining
Trans Image on Processing is reveiewing... After review we will relase the code.
=======
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/AttentionGAN/blob/master/LICENSE.md)


## USIGAN Framework Figure Abstract
The proposed  method introduces the concept of self-information by leveraging optimal transport (OT) and optical density (OD). Through the cycle-consistent transfer strategy of unbalanced optimal transport (UOT) and intra-batch optical density consistency, our approach eliminates weakly paired terms from the joint marginal distribution. Please refer to our papers for more details.
![s](assert/FigureAbstract.png)

USIGAN: USIGAN: Unbalanced Self-Information Feature Transport for Weakly Paired  Image IHC Virtual Staining.<br>
[Yue Peng](http://disi.unitn.it/~hao.tang/)<sup>1,2</sup>, [Bing Xiong](https://scholar.google.com/citations?user=4CQKG8oAAAAJ&hl=en)<sup>1,2</sup>, [Fuqiang Chen](http://www.robots.ox.ac.uk/~danxu/)<sup>1,2</sup>, [De Eybo](https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=en)<sup>1,2</sup>,[RanRan Zhang](http://disi.unitn.it/~sebe/)<sup>2</sup>,[Wanming Hu](http://disi.unitn.it/~sebe/)<sup>1</sup>,[Jing Cai](http://disi.unitn.it/~sebe/)<sup>3</sup> and [Wenjian Qin](http://disi.unitn.it/~sebe/)<sup>1,2</sup>. <br> 
<sup>1</sup>ShenZhen
Institues of Advanced Technology, China.
<sup>2</sup>University chinese academy of sciences, China, <sup>3</sup>Sun Yat-sen University Cancer Cente, China.<sup>4</sup>Department of Health Technology and Informatics, The Hong
Kong Polytechnic University,Hongkong China.<br>
Under Reviewing by Trans Image on Processing. <br>
The repository offers the official implementation of our paper in PyTorch.


## Installation

```bash
conda create -n basicvs python=3.10 -y
conda activate basicvs
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Dataset Preparation
Download the datasets using the following script. Please cite their paper if you use the data. 
The IHC4BC dataset can be download [here](https://ihc4bc.github.io/). The MIST dataset can be download [here](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_61).

## Pretrained Weigths
Download the datasets using the following script. Please cite their paper if you use the data. 
The IHC4BC dataset can be download [here](https://pan.baidu.com/s/17kGl5vxDYclS3NgjLva5VQ?pwd=tip0). The MIST dataset can be download [here]( https://pan.baidu.com/s/1Lpdsq2oCqvutCvm0iWSrKg?pwd=tip0 ).

## Train

```
python -m experiments mist train 0 --gpu_id 0
```

## Evaluation

Note: We change crop size from 512 to 1024, when we are evaluating.
```
python -m experiments mist test 0 --gpu_id 0
```

### Calculate PHV,FID,SSIM,PSNR
```
bash scripts/evaluate_score.sh -r /path/your/results -t /subfile/path/results -e experiment_file_name
```
For example: save results in /root/USIGAN/results/HER2/USIGAN.

you will use bash scripts/evaluate_score.sh -r /root/USIGAN/results -t HER2 -e USIGAN 

### Calculate Content Pearson-R

```
python scripts/eval_pathology_metric.py  --dataroot /path/experiment/results/stain_type  --expname experiment_name
```
### Calculate Pathology Preserve Pearson-R

We need Fiji App to calculate optical density for each image, the details can be viewed in [here](./assert/MetricREADME.md)

```
python scripts/pearson_R.py 
```

## Acknowledgments
This source code is inspired by [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [SIMGAN](https://github.com/xianchaoguan/SIM-GAN).

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([b.xiong@siat.ac.cn](b.xiong@siat.ac.cn)).


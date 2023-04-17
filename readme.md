# Video GNN revised from Vision GNN:
By Kai Han, Yunhe Wang, Jianyuan Guo, Yehui Tang and Enhua Wu. NeurIPS 2022. [[arXiv link]](https://arxiv.org/abs/2206.00272)

![image](./vig.png)

## Requirements
Pytorch 1.7.0,
timm 0.3.2,
torchprofile 0.0.4,
apex

## Training

- Training VideoGNN using 2 GPUs on [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads):
- Download HMDB51 dataset first from the official website
- Move train/test.csv from "data" to the dataset
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2342 train.py --model vig_ti_224_gelu --sched cosine --epochs 110 --opt adamw -j 8 --warmup-lr 1e-6 --model-ema --model-ema-decay 0.99996 --warmup-epochs 10 --decay-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 12 --output ./train_exp --gpu-ids 0,1
```

- Performance: top-1 accuracy is 33.922 for training 110 epoches; 31.307 for training 70 epoches.

## Acknowledgement
This repo partially uses code from [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones).

## Citation
```
@inproceedings{han2022vig,
  title={Vision GNN: An Image is Worth Graph of Nodes}, 
  author={Kai Han and Yunhe Wang and Jianyuan Guo and Yehui Tang and Enhua Wu},
  booktitle={NeurIPS},
  year={2022}
}
```

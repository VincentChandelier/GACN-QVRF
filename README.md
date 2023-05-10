# Learned Focused Plenoptic Image Compression with Microimage Preprocessing and Global Attention + QVRF: A Quantization-error-aware variable rate framework for learned image compression 

Pytorch implementation of the paper "Learned Focused Plenoptic Image Compression with Microimage Preprocessing and Global Attention" with variable rate techniche of "QVRF: A Quantization-error-aware variable rate framework for learned image compression"

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * GACN：https://github.com/VincentChandelier/GACN
 * QVRF：https://github.com/bytedance/QRAF

# AvailableData
 Data |  Link                                                                                              |
| ----|---------------------------------------------------------------------------------------------------|
| FPI2k original images | [FPI2k original images](https://pan.baidu.com/s/1CQ9hKhxY1z-sIHsqr00XXg?pwd=xya8)    |
| Packaged FPI2k original images | [Packaged FPI2k original images](https://pan.baidu.com/s/1UCCqHB0tfEKILJp0cHaucg?pwd=hy0j)    |
| FPI2k preprocessed images| [FPI2k preprocessed images](https://pan.baidu.com/s/1omfnFkK_XQpBrJyE6epkXQ?pwd=2hw0)     |
| Packaged FPI2k preprocessed images| [Packaged FPI2k preprocessed images](https://pan.baidu.com/s/1DkCbAQHN4UP3Cajug3uMjg?pwd=t98i)     |
| TSPC white image | [TSPC white image](https://drive.google.com/file/d/1jaC2OsIWTVjTBicbBOrEr8-T1o4ZuTh0/view?usp=sharing) |
| Training patches | [Training patches](https://pan.baidu.com/s/1hKjg0eXT_bkJfQn8z0z3VA?pwd=p4zm)    |
| ffmpeg | [ffmpeg](https://drive.google.com/file/d/15mvTI74xi4dB3cov7oHByEdARQLAC_XV/view?usp=sharing) |
| Packaged training patches | [Packaged training patches](https://pan.baidu.com/s/1MSn2dEriB1Wal2uOMQe6hg?pwd=daei)    |
| Full-resolution test images | [Full-resolution test images](https://pan.baidu.com/s/1LSFfkxHW1inb04PVt3DwIA?pwd=5lvb) |
| Fixed-rate model checkpoints   | [Model checkpoints](https://pan.baidu.com/s/1hsFpQic6bMRZFvcmbAN7-g?pwd=54rv)|
| Variable rate model checkpoint | [Variable rate model checkpoint](https://drive.google.com/file/d/1ZKmgrAtL6rdYQAoWmiTNUF11E0090Hlr/view?usp=sharing)

## Installation
Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
```bash
conda create -n FPIcompress python=3.9
conda activate FPIcompress
pip install compressai==1.1.5
pip install ptflops
pip install einops
pip install tensorboardX
```
## Usage
### Trainning
stage 1
```
python train.py -d dataset --N 128 --M 192 --depth 2 0 2 0  --heads 4 --dim_head 192 --dropout 0.1 -e 50  -lr 1e-4 -n 8  --lambda 1e-2 --batch-size 4  --test-batch-size 4 --aux-learning-rate 1e-4 --patch-size 384 384 --cuda --save --seed 1926 --gpu-id 0 --savepath  ./checkpoint/GACN --training_stage 1 --stemode 0 --loadFromSinglerate 0
```
stage 2
```
python train.py -d dataset --N 128 --M 192 --depth 2 0 2 0  --heads 4 --dim_head 192 --dropout 0.1 -e 50  -lr 1e-4 -n 8  --lambda 1e-2 --batch-size 4  --test-batch-size 4 --aux-learning-rate 1e-4 --patch-size 384 384 --cuda --save --seed 1926 --gpu-id 0 --savepath  ./checkpoint/GACN_VRNoise --checkpoint ./checkpoint/GACN/checkpoint.pth.tar --training_stage 2 --stemode 0 --loadFromSinglerate 0 --pretrained
```
stage 3
```
python train.py -d dataset --N 128 --M 192 --depth 2 0 2 0  --heads 4 --dim_head 192 --dropout 0.1 -e 20  -lr 1e-6 -n 8  --lambda 1e-2 --batch-size 4  --test-batch-size 4 --aux-learning-rate 1e-4 --patch-size 384 384 --cuda --save --seed 1926 --gpu-id 0 --savepath  ./checkpoint/GACN_VRSTE --checkpoint ./checkpoint/GACN_VRNoise/checkpoint.pth.tar  --training_stage 3 --stemode 1 --loadFromSinglerate 0 --pretrained
```
### Fixed the entropy model
```
python updata.py ./checkpoint/GACN_VRSTE/checkpoint.pth.tar -n GACN_VR_STE
```
### Evaluation
To evaluate a trained model, the evaluation script is:
```bash
python3 Inference.py --dataset ./dataset/FullTest  --s 2 -p ./PLConvSTE.pth.tar --patch 384 --factormode 0 --factor 0
```
More details can refer https://github.com/bytedance/QRAF

## Results
RD results on 20 test images
![Variable rate results of QVRF](asserts/PL2VR.png)


## Citation
```
@article{tong2023qvrf,
  title={QVRF: A Quantization-error-aware Variable Rate Framework for Learned Image Compression},
  author={Tong, Kedeng and Wu, Yaojun and Li, Yue and Zhang, Kai and Zhang, Li and Jin, Xin},
  journal={arXiv preprint arXiv:2303.05744},
  year={2023}
}
```
```
@ARTICLE{10120973,
  author={Tong, Kedeng and Jin, Xin and Yang, Yuqing and Wang, Chen and Kang, Jinshi and Jiang, Fan},
  journal={IEEE Transactions on Multimedia}, 
  title={Learned Focused Plenoptic Image Compression with Microimage Preprocessing and Global Attention}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3272747}}
```

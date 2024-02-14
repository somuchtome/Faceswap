# Face Swap via Diffusion Model

PyTorch implementation of Face Swap via Diffusion Model

## Environment setup
```
cd Faceswap
conda env create -f environment.yaml 
conda activate swap
cd ldm
pip install -e ".[torch]"
```
### note 
the diffusers have been modified to support IP-adapter and our text embedding optimization which is different from huggingface-released and make sure the connection to huggingface is ok

## Download Pretrained Weights
The weights required for FaceParser can be downloaded from [link](
https://gisto365-my.sharepoint.com/personal/hongieee_gm_gist_ac_kr/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhongieee%5Fgm%5Fgist%5Fac%5Fkr%2FDocuments%2FDiffFace%2Fcheckpoints%2FFaceParser%2Epth&parent=%2Fpersonal%2Fhongieee%5Fgm%5Fgist%5Fac%5Fkr%2FDocuments%2FDiffFace%2Fcheckpoints). 
The weights required for ArcFace can be downloaded from [link](https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar). 

```
mkdir checkpoints
mv arcface_checkpoint.tar checkpoints/ 
mv FaceParser.pth checkpoints/ 
```



## Directories structure

The dataset and checkpoints should be placed in the following structures below

```
Faceswap
├── checkpoints
    ├── arcface_checkpoint.tar
    ├── FaceParser.pth
├── data
    └── src
        ├── 001.png
        └── ...
    └── targ
        ├── 001.png
        └── ...
```

## Face swap

Place source and target images in data/src, and data/targ. Then run the following. 

```
sh swap.sh
```

## Face restoration postprocess
```
# git clone this repository
git clone https://github.com/sczhou/CodeFormer.git
cd CodeFormer
python basicsr/setup.py develop
```
### Quick Inference

#### Download Pre-trained Models:
Download the facelib and dlib pretrained models from [[Releases](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) | [Google Drive](https://drive.google.com/drive/folders/1b_3qwrzY_kTQh0-SnBoGBgOrJ_PLZSKm?usp=sharing) | [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/s200094_e_ntu_edu_sg/EvDxR7FcAbZMp_MA9ouq7aQB8XTppMb3-T0uGZ_2anI2mg?e=DXsJFo)] to the `weights/facelib` folder. You can manually download the pretrained models OR download by running the following command:
```
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib (only for dlib face detector)
```

Download the CodeFormer pretrained models from [[Releases](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) | [Google Drive](https://drive.google.com/drive/folders/1CNNByjHDFt0b95q54yMVp6Ifo5iuU6QS?usp=sharing) | [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/s200094_e_ntu_edu_sg/EoKFj4wo8cdIn2-TY2IV6CYBhZ0pIG4kUOeHdPR_A5nlbg?e=AO8UN9)] to the `weights/CodeFormer` folder. You can manually download the pretrained models OR download by running the following command:
```
python scripts/download_pretrained_models.py CodeFormer
```

🧑🏻 Face Restoration (cropped and aligned face)
```
# For cropped and aligned faces (512x512)
python inference_codeformer.py -w 0.5 --has_aligned --input_path [image folder]|[image path]
```


## Acknowledgments
This code borrows heavily from [DiffFace](https://github.com/hxngiee/DiffFace.git), [Diffusers](https://github.com/huggingface/diffusers.git), [Codeformer](https://github.com/sczhou/CodeFormer.git) and [Lora](https://github.com/cloneofsimo/lora.git).


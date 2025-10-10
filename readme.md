<h1 align="center">
  FreqPath-Net
</h1>

<p align="center">
  <img src="img/graph_abstract-FreqPath-Net.jpg" alt="FreqPath-Net" width="100%">
  <br>
  <rm>Figure 1: FreqPath-Net Architecture</rm>
</p>

**📖Title：** FreqPath-Net: Frequency-Aware Spectral–Orthogonal Network for Histopathological Nuclei Segmentation

<!-- **👨‍💻Author：** Jin Huang · Shengqian Wang · Wenchao Xue · Shuo Zhang · Mengping Long · Taobo Hu · Zhaoyi Ye · Sheng Liu (*Fellow, IEEE*) · Du Wang · **Liye Mei** · **Cheng Lei** -->

<!-- **📬 Corresponding Authors**

- **Liye Mei** · [liyemei@whu.edu.cn](mailto:liyemei@whu.edu.cn)  
- **Cheng Lei** · [leicheng@whu.edu.cn](mailto:leicheng@whu.edu.cn)   -->

**📚 Status:** Under review at *IEEE Transactions on Medical Imaging (TMI), 2025* 

**Link：** [![GitHub](https://img.shields.io/badge/GitHub-FreqPath-Net-black?logo=github)](https://github.com/huangjin520/FreqPath-Net) [![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey?logo=readthedocs)]() [![Website](https://img.shields.io/badge/Project-Website-blue?logo=google-chrome)](https://www.lei-whu.com)


**📜Abstract:** <p align="justify"> Nuclei segmentation is a fundamental but challenging task in computational pathology due to diverse morphologies, blurred boundaries, and staining variations. Despite remarkable progress, existing models often suffer from structural instability under morphological and staining variations. We attribute this instability to disrupted frequency–spatial consistency and address it through FreqPath-Net, which enforces frequency–spatial consistency for robust nuclei segmentation. By operating directly in the frequency domain, FreqPath-Net achieves morphology-invariant and stain-robust feature representations. The Spectral Wavelet Attention Module (SWAM) adaptively enhances high-frequency boundary cues while maintaining low-frequency consistency, addressing boundary blurring and detail loss. Furthermore, the Orthogonal Direction-Constrained Frequency Module (ODFM) captures global spectral patterns and enforces directional consistency, effectively preserving boundary orientation and structural integrity by leveraging frequency–spatial consistency. Extensive experiments on twelve nuclei segmentation benchmarks show that FreqPath-Net consistently outperforms state-of-the-art methods. On the PanNuke dataset, FreqPath-Net attains an mIoU of 85.41\%, exceeding the previous best by 1.34\% across diverse organs. The code is available at [FreqPath-Net](https://github.com/huangjin520/FreqPath-Net).

# Introduction
This is an official implementation of [FreqPath-Net: Frequency-Aware Spectral–Orthogonal Network for Histopathological Nuclei Segmentation](). ...



## 🚀 Quick start
### 1️⃣ Installation
Assuming that you have installed PyTorch and TorchVision, if not, please follow the [officiall instruction](https://pytorch.org/) to install them firstly. 
Intall the dependencies using cmd:

``` sh
python -m pip install -r requirements.txt --user -q
```

All experiments use the PyTorch 1.8 framework in a Python 3.10 environment. Other versions of pytorch and Python are not fully tested.
### 📂 Data preparation
We have evaluated segmentation performance on Four nuclei segmentation datasets: 
- [🔬CPM17](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2019.00053/full)  
- [🔬Kumar](https://ieeexplore.ieee.org/abstract/document/7872382)  
- [🔬MoNuSeg](https://ieeexplore.ieee.org/abstract/document/8880654)  
- [🔬PUMA](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf011/8024182) 
- [🔬BBBC020](https://pmc.ncbi.nlm.nih.gov/articles/PMC3627348/) 
- [🔬DSB](https://www.nature.com/articles/s41592-019-0612-7) 
- [🔬CoNSep](https://www.sciencedirect.com/science/article/pii/S1361841519301045) 
- [🔬CPM15](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2019.00053/full) 
- [🔬CryoNuSeg](https://www.sciencedirect.com/science/article/pii/S0010482521001438) 
- [🔬PanNuke](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2) 
- [🔬TNBC](https://ieeexplore.ieee.org/abstract/document/8438559) 
- [🔬NuInsSeg](https://www.nature.com/articles/s41597-024-03117-2) 
Four other modality datasets:
- [🎀BUSI-WHU](https://ieeexplore.ieee.org/abstract/document/10906450)
- [🧠Brain Tumor MRI](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)  
- [📂LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)  
- [👶🏻PSFH](https://www.nature.com/articles/s41597-024-03266-4)  
- [🩺GlaS](https://www.sciencedirect.com/science/article/pii/S1361841516301542)  
- [🎀BACH2018](https://iciar2018-challenge.grand-challenge.org/dataset/)  
- [🎀In house WSI](https://ieeexplore.ieee.org/abstract/document/10752664)  
Dataset tree:
📂 DATASET  
└── 📂 MoNuSeg  
&emsp; ├── 📂 train  
&emsp; │ &emsp; ├── img  
&emsp; │ &emsp; └──  mask  
&emsp; ├── 📂 val  
&emsp; │ &emsp; ├──  img  
&emsp; │ &emsp; └── mask  
&emsp; └── 📂 test  
&emsp; &emsp; ├──img  
&emsp; &emsp; └── mask


### Training
The FreqPath-Net model can be trained on training set using the following: 

```
python train_FreqPath-Net.py 
``` 

The parameters of the model have been carefully designed. 
FreqPath-Net - Hardware: an NVIDIA RTX A6000 GPU and an Intel Core i9-10900X CPU.

## 📊 Evaluation
The FreqPath-Net model can be evaluated on validation set using the following: 

```
python eval.py 
``` 

<p align="center">
  <img src="img/Fig_WSI_info.jpg" alt="FreqPath-Net" width="100%">
  <br>
  <rm>Figure: (a) Workflow of the FreqPath-Net-based WSI analysis pipeline. (b) Tumor burden heatmaps on the public BACH2018 dataset; red contours indicate cancer regions. (c) Predicted nuclei masks on in-house WSIs without ground truth.</rm>
</p>

## 📬 Contact
For any questions or collaborations, please contact [Jin Huang](mailto:jinhuang@whu.edu.cn), [Shengqian Wang](mailto:sqwang@whu.edu.cn) or open an issue on GitHub.



****
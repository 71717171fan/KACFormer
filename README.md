# KACFormer
![图片1](https://github.com/user-attachments/assets/3ab426e5-e6f2-4309-9191-02b7bd093d2e)  
We propose a novel dehazing method that integrates CNNs, Transformers, knowledge transfer, and attention mechanisms, based on Vision Transformers (ViT). Compared to baseline methods and several existing approaches, our method demonstrates superior dehazing performance.  
## Setup
  Pytorch 1.13.1 Intel i5-12600KF CPU NVIDIA GeForce RTX 4090 GPU  
  ### The training process is as follows:  
  First, run train.py under the teacher-network file, with the data path changed to your own path in the exact format shown in DATA.  
  Second, run train.py, where the teacher path is changed to your own trained teacher model.  
  ### The teating process is as follows:  
  Test with test.py.  
  Code Reference Papers"Vision Transformers for Single Image Dehazing"  
## Dataset  
  dataset: The dataset can be found [here](https://pan.baidu.com/s/1-nnKIoMwdVzQmEqiGrW_eQ). extraction code：7i2u  
## Citation


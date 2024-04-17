- Source Code
# Notice
- a Due to some problems I can only use .py files. You can still use ipnyb files if you want
- Rename the parquet files to train, test0 and test1 and put them into your directory before running any code.
- Make sure to install all packages, or run the code on online servers.
- I strongly recommend to run the code locally. Y'all know how bad colab is.
# How to Run on your device
- Required python versioin: 3.8 or higher. 
- Install all necessary packages on your own device.
- You will need to install pytorch for gpu on your computer. This requires an Nvidia GPU as well as cuda and cudnn installed.
- cuda: https://developer.nvidia.com/cuda-12-1-0-download-archive
- cudnn https://developer.nvidia.com/cudnn-downloads
- install pytorch use this pip command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- After installation, you should check if cuda is available. In python, type import torch, and see if torch.cuda.is_available() is true
- Other required packages: matplotlib, pandas, numpy, tqdm, PIL( pip install pillow), pycharm, torchmetrics, sklearn 
- Download the train****.parquet file from the dataset.
- use pandas and sklearn to split the dataset into 3 parts. train, validation and test, ratio .8:.1:.1. Name these files as shown in the train.py and test.py
- Run the train.py code for training, test.py for testing. Note that the model file is not in the repo, either train it yourself or ask me.
- 来自Hank: 是不是应该请我吃顿饭哈哈哈哈哈

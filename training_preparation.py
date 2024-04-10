import PIL
import io
import pandas as pd
from PIL import Image
ALL_CHAR = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
MAX_WIDTH= 589
MAX_HEIGHT = 378
def binarize_image_from_binary(data):
    image = Image.open(io.BytesIO(data['bytes']))
    grayscale = image.convert('L')  
    binary = grayscale.point(lambda x: 0 if x < 128 else 255, '1')  # Convert to binary
    return binary
def resize(image, target_width, target_height):
    result = Image.new('1', (target_width, target_height), 255)
    padding = ((target_width - image.width) // 2, (target_height - image.height) // 2)
    result.paste(image, padding)
    return result
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as transform
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform = None, target_transform = None):
        self.df = pd.read_parquet(file_path, engine='pyarrow')
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        image = self.df.at[idx, 'image']
        image = binarize_image_from_binary(image)
        image = resize(image, MAX_WIDTH, MAX_HEIGHT)
        label = self.df.at[idx, 'text']
        return image, label


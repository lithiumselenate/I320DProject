import PIL
import io
import os
import pandas as pd
from PIL import Image
# Get resize parameter for training
def binarize_image_from_binary(data):
    image = Image.open(io.BytesIO(data['bytes']))
    grayscale = image.convert('L')  
    binary = grayscale.point(lambda x: 0 if x < 128 else 255, '1') 
    return binary
df_train = pd.read_parquet('train.parquet', engine='pyarrow')
'''df_test1 = pd.read_parquet('test1.parquet', engine='pyarrow')
df_test0 = pd.read_parquet('test0.parquet', engine='pyarrow')
print(df_train.iloc[0])
df_train['processed_image'] = df_train['image'].apply(binarize_image_from_binary)
df_test1['processed_image'] = df_test1['image'].apply(binarize_image_from_binary)
df_test0['processed_image'] = df_test0['image'].apply(binarize_image_from_binary)
max_width = max(df_train['processed_image'].apply(lambda img: img.width))
max_height = max(df_train['processed_image'].apply(lambda img: img.height))
max_width = max_width  if max_width > max(df_test0['processed_image'].apply(lambda img: img.width)) else max(df_test0['processed_image'].apply(lambda img: img.width))
max_width = max_width  if max_width > max(df_test1['processed_image'].apply(lambda img: img.width)) else max(df_test1['processed_image'].apply(lambda img: img.width))
max_height = max_height  if max_height > max(df_test0['processed_image'].apply(lambda img: img.height)) else max(df_test0['processed_image'].apply(lambda img: img.height))
max_height = max_height  if max_height > max(df_test1['processed_image'].apply(lambda img: img.height)) else max(df_test1['processed_image'].apply(lambda img: img.height))
print(max_width)
print(max_height)'''
#Get dictionary for all the characters in the dataset
crcset = set()
for text in df_train['text']:
    crcset.update(text)
str1 = ''.join(sorted(crcset))
print(str1)




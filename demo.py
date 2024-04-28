import PIL
from PIL import Image
import torch
import torchvision
import training_preparation as tr
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
ALL_CHAR = ' -ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
def create_char_dict(s):
    return {c: i for c, i in enumerate(s)}
char_dict = create_char_dict(ALL_CHAR)
model = tr.CTCModel(chan_in=1,                                                           
                 time_step=17,                                                                                                  
                 feature_size=512,                                                     
                 hidden_size=512,                        
                 output_size=len(char_dict),                               
                 num_rnn_layers=4,                                                      
                 rnn_dropout=0) 
model.load_state_dict(torch.load('new_model_state_dict.pth'))
model.to('cpu')
model.eval()
# Read the image
'''
image = Image.open("test5.png")

# Binarize the image
threshold = 128
binary_image = image.convert("L").point(lambda pixel: 0 if pixel < threshold else 255)

# Transform the binary image into a torch tensor
transform = torchvision.transforms.ToTensor()
tensor_image = transform(binary_image)
print(tensor_image.shape)
print()
# Add a batch dimension
tensor_image = tensor_image.unsqueeze(0)
logit = model(tensor_image)
logit = logit.argmax(2).squeeze().cpu().numpy()
print(logit)
predicted = logit
predicted = predicted[predicted != 0]
print(predicted)
predicted = [x for i, x in enumerate(predicted) if i == 0 or x != 1 or predicted[i-1] != 1]
print(char_dict)
chars = [char_dict[c] for c in predicted]
p = ''.join(chars)
print(p)
'''
import pandas as pd
def get_single(image):
    threshold = 128
    binary_image = image.convert("L").point(lambda pixel: 0 if pixel < threshold else 255)
    data = {'image': [image], 'binary_image': [binary_image], 'text': ['']}
    df = pd.DataFrame(data)
    dataset = tr.DemoDataset(df = df,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=tr.collate_fn_test_cpu)
    for images, targets in dataloader:
        logit = model(images)
        logit = logit.argmax(2).squeeze().cpu().numpy()
        predicted = logit
            
        predicted = predicted[predicted != 0]
        predicted = [x for i, x in enumerate(predicted) if i == 0 or x != 1 or predicted[i-1] != 1]
        chars = [dataset.char_dict[c] for c in predicted]
        p = ''.join(chars)
        return p

    '''
    transform = torchvision.transforms.ToTensor()
    tensor_image = transform(binary_image)
    tensor_image = tensor_image.unsqueeze(0)
    logit = model(tensor_image)
    logit = logit.argmax(2).squeeze().cpu().numpy()
    predicted = logit
    predicted = predicted[predicted != 0]
    predicted = [x for i, x in enumerate(predicted) if i == 0 or x != 1 or predicted[i-1] != 1]
    chars = [char_dict[c] for c in predicted]
    p = ''.join(chars)
    return p
    '''
import gradio as gr
import demo 
import training_preparation

inputs=gr.Image(label="Input Image", type="pil")
output1 = gr.Textbox(label="Prediction")

gr.Interface(
    fn=demo.get_single,
    inputs=inputs,
    outputs=output1,
    title="Handwriting recognition",
    description="Select File and click on submit to get the prediction."
).launch(share=True)
# tests.py contains tests for the code 

import torch
from ViT_model import Vision_Transformer_Pretraining, Vision_Transformer_Finetuning
from OxfordPets_Data import Oxford_Pets_data_setUp

#Oxford Pets Dataset Test
label_list, train_data, test_data = Oxford_Pets_data_setUp()
print('Oxford Pets train image shape test') 
assert train_data[0][0].shape == torch.Size([3, 384, 384]), 'Image shape should be [3, 384, 384]'
print('TEST PASSED: training image has shape [3, 384, 384]')
print('Oxford Pets test image shape test') 
assert test_data[0][0].shape == torch.Size([3, 384, 384]), 'Image shape should be [3, 384, 384]'
print('TEST PASSED: testing image has shape [3, 384, 384]')    

#Test Vision_Transformer_Pretraining using default values (see Vision_Transformer_Pretraining function in ViT_model.py for values)
batch_size = 4
image = torch.rand([batch_size,3,224,224])
model = Vision_Transformer_Pretraining()
output = model(image)
print('Vision_Transformer_Pretraining model output shape test:')
assert output.shape == torch.Size([batch_size,1000]), 'Output shape should be [{batch_size}, 1000]'
print(f'TEST PASSED: output has shape [{batch_size}, 1000]')

#Test Vision_Transformer_Finetuning using default values (see Vision_Transformer_Finetuning function in ViT_model.py for values)
image2 = torch.rand([batch_size,3,384,384])
model2 = Vision_Transformer_Finetuning()
output2 = model2(image2)
print('Vision_Transformer_Finetuning model for Oxford Pets Dataset output shape test:')
assert output2.shape == torch.Size([batch_size,37]), 'Output shape should be [{batch_size}, 37]'
print(f'TEST PASSED: output has shape [{batch_size}, 37]')
# finetuning_test.py contains testing pipeline for a finetuned ViT model
# ViT is the Vision Transformer model introduced in
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitsky et al. 2021
# arxiv: https://arxiv.org/abs/2010.11929
# This is a paper replication carried out to improve my programming skills 
# Author: Carissa Cullen 

import torch
from OxfordPets_Data import Oxford_Pets_data_setUp
from training_testing_loops import test

#Create Oxford Pets datasets
label_list, train_data, test_data = Oxford_Pets_data_setUp() 

#Load trained model if necessary
finetuning_model = torch.load('ViT_Paper_Replication/vitB16_ImageNet1K,finetuned_model.pt', weights_only=False)

#Create device agnostic code. Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
finetuning_model.to(device)

#I found the loss used in line 48 of vision_transformer/vit_jax/train.py (from original code repo)
loss_fn = torch.nn.CrossEntropyLoss()

test(finetuning_model, test_data, loss_fn)


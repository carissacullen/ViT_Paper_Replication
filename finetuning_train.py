# finetuning_train.py contains training pipeline for finetuning a ViT model
# ViT is the Vision Transformer model introduced in
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitsky et al. 2021
# arxiv: https://arxiv.org/abs/2010.11929
# This is a paper replication carried out to improve my programming skills 
# Author: Carissa Cullen 

import torchvision
import torch
from torch.utils.data import DataLoader
from OxfordPets_Data import Oxford_Pets_data_setUp
from ViT_model import Vision_Transformer_Pretraining, Vision_Transformer_Finetuning
from transfer_pretraining_to_finetuning import vitB16_model_to_pretraining_model, initialise_finetuning_parameters
from training_testing_loops import train

#Create Oxford Pets datasets
label_list, train_data, test_data = Oxford_Pets_data_setUp() 

#Using vitB16 model from torchvision.models
vitB16_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
vitB16_model = torchvision.models.vit_b_16(weights=vitB16_weights)

#Initialise a random pretrained model
model = Vision_Transformer_Pretraining()

#Transfer weights from vitB16 to randomly initialiseed model (Don't need this step
# if you train a pretrained model yourself with Vision_Transformer_Pretraining)
pretrained_model = vitB16_model_to_pretraining_model(vitB16_model,model)

#Initialise a random finetuning model
finetuning_model = Vision_Transformer_Finetuning()

#Transfer weights from pretrained model to finetuining model
finetuning_model = initialise_finetuning_parameters(finetuning_model,pretrained_model)

#Empty GPU memory 
torch.cuda.empty_cache() 

#Create training dataloader. The true batch size is 512 and can be found under Appendix B.1.1 - Finetuning (page 13)
#However, with the A100 GPU I could only use batch size 8 due to compute limitations
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

#Create device agnostic code. Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
finetuning_model.to(device)

#I found the loss used in line 48 of vision_transformer/vit_jax/train.py (from original code repo)
loss_fn = torch.nn.CrossEntropyLoss()
#Hyperparameters found in Table 3 and Section 4.1 of the ViT paper (Dosovitskiy et al. 2021)
optimizer = torch.optim.SGD(finetuning_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500*len(train_dataloader), 1e-5) 

#Used 70 epochs as this is the right number of epochs to use if you want 500 steps with a batch_size of 512
#train_data_length/batch_size = 3680/512 = 7.1875 steps per epoch. To get 500 steps you need 70 epochs  
epochs = 70

train(finetuning_model, train_dataloader, loss_fn, optimizer, epochs, scheduler)

#Save the model to enable running a test loop without training the model each time 
torch.save(finetuning_model, 'ViT_Paper_Replication/vitB16_ImageNet1K,finetuned_model.pt')
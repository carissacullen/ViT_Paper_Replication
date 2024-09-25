# transfer_pretraining_to_finetuning.py containes functions needed to transfer the parameters of a pretrained ViT model to
# a finetuning model. 
# ViT is the Vision Transformer model introduced in
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitsky et al. 2021
# arxiv: https://arxiv.org/abs/2010.11929
# This is a paper replication carried out to improve my programming skills 
# Author: Carissa Cullen 

import torch
from math import sqrt
from scipy.ndimage import zoom

def interpolate_position_embedding(pretrained_embeddings,pretraining_resolution=224,finetuning_resolution=384,patch_size=16,latent_vector_size=768):
  '''
  Performs 2D interpolation of position embeddings of pretrained model
  to allow for increased resolution of images from pretraining to finetuning
  Explanation can be found in section 3.2 of Dosovitsky et al. 2021
  '''
  pretraining_num_patches = int((pretraining_resolution/patch_size)**2)
  finetuning_num_patches = int((finetuning_resolution/patch_size)**2)
  sqrt_pretraining_num_patches = int(sqrt(pretraining_num_patches))
  sqrt_finetuning_num_patches = int(sqrt(finetuning_num_patches))
  #Remove class embedding from pretrainined embeddings tensor
  old_position_embeddings = pretrained_embeddings[1:]
  #Reshape into a grid so that each patch embedding has 2D positional info
  old_position_embeddings = torch.reshape(old_position_embeddings,[sqrt_pretraining_num_patches,sqrt_pretraining_num_patches,latent_vector_size])
  #Increase grid size to allow for increased resolution of images in finetuning
  #and interpolate missing values
  scaling = sqrt_finetuning_num_patches/sqrt_pretraining_num_patches
  position_interpolated = zoom(old_position_embeddings.detach().numpy(),(scaling,scaling,1))
  position_interpolated = torch.tensor(position_interpolated)
  #Flatten the grid
  new_position_embedding = torch.reshape(position_interpolated,[int(finetuning_num_patches),latent_vector_size])
  #Prepend class embedding to new position embeddings
  new_position_embedding = torch.cat([pretrained_embeddings[0].unsqueeze(0),new_position_embedding],dim=0)

  return new_position_embedding

def vitB16_model_to_pretraining_model(pretrained_model, initialised_model):
  '''
  Transfers the model weights from the pretrained model vitB16 to an initialised pretraining model
  The output therefore has the right parameter names to be fed into Vision_Transformer_Finetuning
  '''
  initialised_model.embeddings.class_embedding = torch.nn.parameter.Parameter(pretrained_model.class_token.squeeze().squeeze())
  initialised_model.embeddings.position_embeddings = torch.nn.parameter.Parameter(pretrained_model.encoder.pos_embedding.squeeze())
  initialised_model.embeddings.linear_projection.weight = torch.nn.parameter.Parameter(pretrained_model.conv_proj.weight.flatten(1,3))
  initialised_model.embeddings.linear_projection.bias = torch.nn.parameter.Parameter(pretrained_model.conv_proj.bias)
  for i in range(12):
    initialised_model.Transformer[i].layer_norm.weight = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].ln_1.weight)
    initialised_model.Transformer[i].layer_norm.bias = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].ln_1.bias)
    initialised_model.Transformer[i].MSA_module.in_proj_weight = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].self_attention.in_proj_weight)
    initialised_model.Transformer[i].MSA_module.in_proj_bias = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].self_attention.in_proj_bias)
    initialised_model.Transformer[i].MSA_module.out_proj.weight = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].self_attention.out_proj.weight)
    initialised_model.Transformer[i].MSA_module.out_proj.bias = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].self_attention.out_proj.bias)
    initialised_model.Transformer[i].layer_norm_2.weight = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].ln_2.weight)
    initialised_model.Transformer[i].layer_norm_2.bias = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].ln_2.bias)
    initialised_model.Transformer[i].MLP_module[0].weight = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].mlp[0].weight)
    initialised_model.Transformer[i].MLP_module[0].bias = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].mlp[0].bias)
    initialised_model.Transformer[i].MLP_module[3].weight = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].mlp[3].weight)
    initialised_model.Transformer[i].MLP_module[3].bias = torch.nn.parameter.Parameter(pretrained_model.encoder.layers[i].mlp[3].bias)
  initialised_model.layer_norm.weight = torch.nn.parameter.Parameter(pretrained_model.encoder.ln.weight)
  initialised_model.layer_norm.bias = torch.nn.parameter.Parameter(pretrained_model.encoder.ln.bias)

  return initialised_model

def initialise_finetuning_parameters(finetuning_model, pretrained_model, pretraining_image_resolution=224,finetuning_image_resolution=384,patch_size=16,latent_vector_size=768):
  '''
  Transfers the model weights from the pretrained model to the finetuning model
  '''
  finetuning_model.embeddings.class_embedding = pretrained_model.embeddings.class_embedding
  #interpolate the position embeddings from the pretraining to finetuning image resolution version
  interpolated = interpolate_position_embedding(pretrained_model.embeddings.position_embeddings,pretraining_image_resolution,finetuning_image_resolution,patch_size,latent_vector_size)
  finetuning_model.embeddings.position_embeddings = torch.nn.parameter.Parameter(interpolated)
  finetuning_model.embeddings.linear_projection.weight = pretrained_model.embeddings.linear_projection.weight
  finetuning_model.embeddings.linear_projection.bias = pretrained_model.embeddings.linear_projection.bias
  for i in range(12):
    finetuning_model.Transformer[i].layer_norm.weight = pretrained_model.Transformer[i].layer_norm.weight
    finetuning_model.Transformer[i].layer_norm.bias = pretrained_model.Transformer[i].layer_norm.bias
    finetuning_model.Transformer[i].MSA_module.in_proj_weight = pretrained_model.Transformer[i].MSA_module.in_proj_weight
    finetuning_model.Transformer[i].MSA_module.in_proj_bias = pretrained_model.Transformer[i].MSA_module.in_proj_bias
    finetuning_model.Transformer[i].MSA_module.out_proj.weight = pretrained_model.Transformer[i].MSA_module.out_proj.weight
    finetuning_model.Transformer[i].MSA_module.out_proj.bias = pretrained_model.Transformer[i].MSA_module.out_proj.bias
    finetuning_model.Transformer[i].layer_norm_2.weight = pretrained_model.Transformer[i].layer_norm_2.weight
    finetuning_model.Transformer[i].layer_norm_2.bias = pretrained_model.Transformer[i].layer_norm_2.bias
    finetuning_model.Transformer[i].MLP_module[0].weight = pretrained_model.Transformer[i].MLP_module[0].weight
    finetuning_model.Transformer[i].MLP_module[0].bias = pretrained_model.Transformer[i].MLP_module[0].bias
    finetuning_model.Transformer[i].MLP_module[3].weight = pretrained_model.Transformer[i].MLP_module[3].weight
    finetuning_model.Transformer[i].MLP_module[3].bias = pretrained_model.Transformer[i].MLP_module[3].bias
  finetuning_model.layer_norm.weight = pretrained_model.layer_norm.weight
  finetuning_model.layer_norm.bias = pretrained_model.layer_norm.bias

  return finetuning_model
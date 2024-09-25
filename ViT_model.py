# ViT_model.py containes the pytorch modules needed for the Vision Transformer architecture as described in
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitsky et al. 2021
# arxiv: https://arxiv.org/abs/2010.11929
# This is a paper replication carried out to improve my programming skills 
# Author: Carissa Cullen 

import torch

#Equivalent to equation 1 in Dosovitsky et al. 2021 (explanation found in section 3.1 of the paper)
class Embeddings_Set_Up(torch.nn.Module):
  '''
  Class that performs the embeddings set-up for the ViT model in Dosovitskiy et al. 2021

  Input: image or batch of images as tensor. [batch_size, num_of_colour_channels, height, width]
         [batch_size, 3, 224, 224] for pretraining, [batch_size, 3, 384, 384] for finetuning

  Output: embeddings tensor of shape [batch_size, num_patches+1, latent_vector_size]
          [batch_size, 197, 768] for pretraining, [batch_size, 577, 768] for finetuning
  '''
  def __init__(self, image_resolution, num_image_channels, patch_size, latent_vector_size):
    super().__init__()
    self.unfold = torch.nn.Unfold(kernel_size=(patch_size,patch_size), stride=(patch_size,patch_size))
    self.linear_projection = torch.nn.Linear((patch_size**2)*num_image_channels,latent_vector_size,bias=True)
    self.class_embedding = torch.nn.parameter.Parameter(torch.randn(latent_vector_size))
    self.position_embeddings = torch.nn.parameter.Parameter(torch.randn(int((image_resolution/patch_size)**2+1),latent_vector_size))

  def forward(self, images):
    #Step 1 - Turn image into patches and flatten them
    image_patches = self.unfold(images)
    image_patches = image_patches.transpose(1,2)
    #Step 2 - Map each patch to a vector of length D (latent vector size)
    patch_embeddings = self.linear_projection(image_patches)
    #Step 3 - Prepend a learnable embedding for the image class to the sequence of patches
    unsqueezed_class_embedding = self.class_embedding.unsqueeze(0).unsqueeze(0)
    batched_class_embeddings = torch.cat([unsqueezed_class_embedding for i in range(patch_embeddings.shape[0])],dim=0)
    class_and_patch_embeddings = torch.cat([batched_class_embeddings,patch_embeddings],dim=1)
    #Step 4 - Add learnable position embeddings to the class and patch embeddings.
    position_embeddings = self.position_embeddings.unsqueeze(0)
    batched_position_embeddings = torch.cat([position_embeddings for i in range(patch_embeddings.shape[0])],dim=0)
    transformer_input = class_and_patch_embeddings + batched_position_embeddings

    return transformer_input
  
#Equivalent to equations 2 & 3 of Dosovitsky et al. 2021
class Transformer_Layer(torch.nn.Module):
  '''
  Transformer Layer for ViT model from Dosovitskiy et al. 2021

  Input: embeddings tensor of shape [batch_size, num_patches+1, latent_vector_size]

  Output: embeddings tensor of shape [batch_size, num_patches+1, latent_vector_size]
  '''
  def __init__(self, num_patches, latent_vector_size, num_MSA_heads, MLP_hidden_layer_size, dropout=0.1):
    super().__init__()

    #Layer Normalisation
    self.layer_norm = torch.nn.LayerNorm(latent_vector_size)
    #Multi-headed Self Attention (MSA)
    self.MSA_module = torch.nn.MultiheadAttention(latent_vector_size, num_MSA_heads, batch_first=True)

    #Layer Normalisation
    self.layer_norm_2 = torch.nn.LayerNorm(latent_vector_size)
    #Multi-Layer Perceptron (MLP)
    self.MLP_module = torch.nn.Sequential(
        torch.nn.Linear(latent_vector_size,MLP_hidden_layer_size),
        torch.nn.Dropout(dropout), #dropout after every dense layer (from Appendix B.1 - 'Training')
        torch.nn.GELU(),
        torch.nn.Linear(MLP_hidden_layer_size,latent_vector_size),
        torch.nn.Dropout(dropout) #dropout after every dense layer (from Appendix B.1 - 'Training')
    )

  def forward(self, transformer_input):
    layer_norm = self.layer_norm(transformer_input)
    MSA_module_output, MSA_weights = self.MSA_module(layer_norm,layer_norm,layer_norm) 
    z_apostraphe = MSA_module_output + transformer_input #Residual Connection
    transformer_output = self.MLP_module(self.layer_norm_2(z_apostraphe)) + z_apostraphe #Residual Connection

    return transformer_output

class Vision_Transformer_Pretraining(torch.nn.Module):
  '''
  Replication of Vision Transformer model from Dosovitskiy et al. (2021)
  Pretraining version where classification head is a MLP with one hidden layer

  Input: image or batch of images as tensor. [batch_size, num_of_colour_channels, height, width]
         for default values, the input size is [batch_size, 3, 224, 224]

  Output: classification probabilities tensor of shape [batch_size, num_labels]
          for Oxford Pets Dataset output size is [batch_size, 37]
  '''
  def __init__(self, image_resolution=224, num_image_channels=3, num_labels=1000, patch_size=16, latent_vector_size=768, num_transformer_layers=12,
               num_MSA_heads=12,MLP_hidden_layer_size=3072,dropout=0.0):
    super().__init__()

    # Dropout
    self.dropout = torch.nn.Dropout(dropout)
    # Create patch embeddings to feed into Transformer block
    self.embeddings = Embeddings_Set_Up(image_resolution, num_image_channels, patch_size, latent_vector_size)
    # Create a Trarnsformer block with L layers, L = 'num_transformer_layers'
    self.Transformer = torch.nn.ModuleList([Transformer_Layer(int((image_resolution/patch_size)**2), latent_vector_size, num_MSA_heads, MLP_hidden_layer_size,dropout) for i in range(num_transformer_layers)])
    #Final Layer Normalisation
    self.layer_norm = torch.nn.LayerNorm(latent_vector_size)
    # Create MLP classification head
    self.classification_head = torch.nn.Sequential(
        torch.nn.Linear(latent_vector_size,MLP_hidden_layer_size),
        torch.nn.Dropout(dropout), #dropout after every dense layer (from Appendix B.1 - 'Training')
        torch.nn.Tanh(),
        torch.nn.Linear(MLP_hidden_layer_size,num_labels),
        torch.nn.Dropout(dropout), #dropout after every dense layer (from Appendix B.1 - 'Training')
    )

  def forward(self, images):
    x = self.embeddings(images)
    x = self.dropout(x)
    for layer in self.Transformer:
      x = layer(x)
    x = self.layer_norm(x)
    y = self.classification_head(x[:,0,:]) #Classification layer only applies to class embedding
    return y    
  
class Vision_Transformer_Finetuning(torch.nn.Module):
  '''
  Replication of Vision Transformer model from Dosovitskiy et al. (2021)
  Finetuning version where classification head is replaced with a linear layer

  Input: image or batch of images as tensor. [batch_size, num_of_colour_channels, height, width]
         for default values, the input size is [batch_size, 3, 384, 384]

  Output: classification probabilities tensor of shape [batch_size, num_labels]
          for Oxford Pets Dataset output size is [batch_size, 37]
  '''
  def __init__(self, image_resolution=384, num_image_channels=3, num_labels=37, patch_size=16, latent_vector_size=768, num_transformer_layers=12,
               num_MSA_heads=12,MLP_hidden_layer_size=3072,dropout=0.1):
    super().__init__()

    # Dropout
    self.dropout = torch.nn.Dropout(dropout)
    # Create patch embeddings to feed into Transformer block
    self.embeddings = Embeddings_Set_Up(image_resolution, num_image_channels, patch_size, latent_vector_size)
    # Create a Trarnsformer block with L layers, L = 'num_transformer_layers'
    self.Transformer = torch.nn.ModuleList([Transformer_Layer(int((image_resolution/patch_size)**2), latent_vector_size, num_MSA_heads, MLP_hidden_layer_size,dropout) for i in range(num_transformer_layers)])
    #Final Layer Normalisation
    self.layer_norm = torch.nn.LayerNorm(latent_vector_size)
    # Create Linear classification layer to apply to class embedding
    self.classification = torch.nn.Linear(latent_vector_size,num_labels)
    # Initialise Classification Linear Layer weight
    torch.nn.init.zeros_(self.classification.weight)
    # Initialise Classification Linear Layer bias
    torch.nn.init.zeros_(self.classification.bias)


  def forward(self, images):
    x = self.embeddings(images)
    x = self.dropout(x)
    for layer in self.Transformer:
      x = layer(x)
    x = self.layer_norm(x)
    y = self.classification(x[:,0,:]) # Classification layer only applies to class embedding
    y = self.dropout(y)
    return y  
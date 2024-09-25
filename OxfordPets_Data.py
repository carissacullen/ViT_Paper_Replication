# OxfordPets_Data.py contains functions needed to prepare the Oxford IIIT Pets dataset to use in the finetuning of ViT models.
# Find dataset on HuggingFace: https://huggingface.co/datasets/timm/oxford-iiit-pet
# ViT is the Vision Transformer model introduced in
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitsky et al. 2021
# arxiv: https://arxiv.org/abs/2010.11929
# This is a paper replication carried out to improve my programming skills 
# Author: Carissa Cullen 

from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset

ds = load_dataset("timm/oxford-iiit-pet")

def create_label_list(ds):
    '''
    Creates a list of label names for the Oxford Pets dataset
    label_names[0] reutrns the breed name associated with label 0
    '''
    #Creates a list of the indexes of the first instance of each label
    indexes = [ds['train']['label'].index(x) for x in set(ds['train']['label'])]
    label_names = []
    #Iterates through the unique labels and returns the label_id name with numbers removed from the end. This leaves the breed name
    for i in range(len(indexes)):
        unique_image_id = ds['train']['image_id'][indexes[i]]
        cleaned = unique_image_id.rstrip('0123456789_')
        label_names.append(cleaned)
    return label_names

data_transform = transforms.Compose([
    # Resize the images to 448x448
    transforms.Resize(size=(448, 448)),
    # Randomly crops the images to 384x384
    transforms.RandomCrop(size=(384, 384)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turns the images into torch.Tensor format
    transforms.ToTensor(),
    #Converts any images in RGBA format to RGB
    transforms.Lambda(lambda x: x[:3]),
])

class OxfordPets(Dataset):

    def __init__(self, dataset=ds['train'], transform=None) -> None:

        # Setup transforms
        self.transform = transform
        # Create images and labels attribute
        self.images = dataset['image']
        self.labels = dataset['label']

    #Function to load images
    def load_image(self, index: int):
        '''Opens an image'''
        im = self.images[index]
        return im

    #Overwrites the __len__() method
    def __len__(self) -> int:
        '''Returns the total number of samples'''
        return len(self.images)

    #Overwrites the __getitem__() method
    def __getitem__(self, index: int):
        '''Returns one sample of data, data and label (X, y)'''
        im = self.images[index]
        label = self.labels[index]

        # Transform if necessary
        if self.transform:
            return self.transform(im), label # return data, label (X, y)
        else:
            return im, label # return data, label (X, y)
        
def Oxford_Pets_data_setUp():
    ds = load_dataset("timm/oxford-iiit-pet")

    label_names = create_label_list(ds)

    train_data = OxfordPets(ds['train'],data_transform)
    test_data = OxfordPets(ds['test'],data_transform) 

    return label_names, train_data, test_data
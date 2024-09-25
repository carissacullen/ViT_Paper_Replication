# training_testing_loops.py contains the train loop and test loop for ViT models.
# ViT is the Vision Transformer model introduced in
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitsky et al. 2021
# arxiv: https://arxiv.org/abs/2010.11929
# This is a paper replication carried out to improve my programming skills 
# Author: Carissa Cullen 

import torch
from tqdm import tqdm

#Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_dataloader, loss_fn, optimizer, epochs, scheduler):

  for epoch in range(epochs):
    train_loss = 0
    correct = 0
    train_accuracy = 0

    for batch, (X, y) in enumerate(train_dataloader):
      #Move data to device
      X, y = X.to(device), y.to(device)

      # Calculate the prediction error
      pred_probabilities = model(X)
      pred = pred_probabilities.argmax(1)
      loss = loss_fn(pred_probabilities, y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #For ImageNet Pretraining and all Finetuning
      optimizer.step()
      scheduler.step()

      #Updating the number of correct predictions
      for i in range(len(pred)):
        if pred[i] == y[i]:
          correct += 1

      train_loss += loss.item()

    #Average training loss over all batches (for one epoch)
    train_loss = train_loss/len(train_dataloader)
    #Avarage Training accuracy over one iteration through the data (for one epoch)
    train_accuracy = correct/len(train_dataloader.dataset) * 100

    print(f'For epoch: {epoch} - Training Loss: {train_loss} - Training Accuracy: {train_accuracy}')

def test(model, test_data, loss_fn):
  model.eval()
  model.to(device)

  with torch.inference_mode():
    test_loss = 0
    correct = 0
    test_accuracy = 0

    for i in tqdm(range(len(test_data))):
      X, y = test_data[i][0], test_data[i][1]
      X = X.unsqueeze(0)
      X, y = torch.tensor(X).to(device), torch.tensor([y]).to(device)
      # Calculate the prediction error
      pred_probabilities = model(X)
      pred = pred_probabilities.argmax(1)
      loss = loss_fn(pred_probabilities, y)

      if pred == y:
        correct += 1

      test_loss += loss.item()

    #Average test loss over batches
    test_loss = test_loss/len(test_data)
    #Test accuracy for batch
    test_accuracy = correct/len(test_data) * 100

    print(f'Testing Loss: {test_loss} - Testing Accuracy: {test_accuracy}')    
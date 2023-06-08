import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

import argparse


import data_setup, engine

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs",type=int,default=5, 
                    help="Number of epochs to train the model for.")


args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the best available weights for the model to be used in transfer learning.
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Get the transforms used for the pretrained model.

transforms = weights.transforms()

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
                                                                train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                        transform=transforms,
                                                                        batch_size=BATCH_SIZE)

# Define the model

model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# freeze the parameters of the model in the features section.
for params in model.features.parameters():
    params.requires_grad = False

output_shape = len(class_names)

model.classifier = nn.Sequential(
                    torch.nn.Dropout(p=0.2, inplace=False),
                    torch.nn.Linear(in_features=1280,
                                     out_features=output_shape,
                                       bias=True)).to(device)


loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

results = engine.train_model(model=model,
                        train_loader=train_dataloader,
                        test_loader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        num_epochs=NUM_EPOCHS)










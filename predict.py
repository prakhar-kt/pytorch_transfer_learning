import torch
from torch import nn
import argparse
import torchvision

parser = argparse.ArgumentParser()

parser.add_argument("--image_path",type=str,required=True,)

args = parser.parse_args()

image_path = args.image_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

transforms = weights.transforms()


class_names = ["pizza","steak","sushi"]

output_shape = len(class_names)



# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True))

model.load_state_dict(torch.load("models/efficientnet_b0.pth"))



def predict(model: nn.Module,
            image_path: str = image_path,
            transforms=transforms):
    
    '''
    Predicts the class of an image using a trained PyTorch model.
    '''

    model.to(device)
    

    image = torchvision.io.read_image(image_path).type(torch.float32)

    model.eval()

    with torch.inference_mode():

        tr_image = transforms(image).unsqueeze(0)



        pred = model(tr_image.to(device))

        pred_probs = torch.nn.functional.softmax(pred,dim=1)
        pre_label = torch.argmax(pred_probs,dim=1)

        print(f"Predicted class: {class_names[pre_label]}")

if __name__ == "__main__":
    predict(model=model,
            image_path=image_path,
            transforms=transforms)












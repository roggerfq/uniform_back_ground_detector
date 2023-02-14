import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import cv2
from PIL import Image


class Classifier():
    def __init__(self, path_model):
        self.model = torchvision.models.resnet18(pretrained=False)
        self.num_ftrs = self.model.fc.in_features #resnet
        self.model.fc = nn.Linear(self.num_ftrs, 2) #3 categorias
        self.model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
        self.model.eval()

        ##transform###
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                #transforms.RandomRotation((0, 360), expand=False),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        #############
    
    @torch.no_grad()
    def evaluate(self, patch):
        height, width, _ = patch.shape
        if(height != 224 or width != 224):
           dsize = (224, 224)#tamaño entrada cnn
           patch = cv2.resize(patch, dsize, interpolation = cv2.INTER_NEAREST)
        patch = patch[:,:,::-1]
        input_tensor = self.transform(Image.fromarray(patch)).unsqueeze(0)
        output = self.model(input_tensor)
        _, pred = torch.max(output, 1)
        return pred.item()


path_model = './models/model_val_acc_99.pth'
classifier = Classifier(path_model)





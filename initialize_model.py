"""## **5. Model Development: using pre-trained modules**
### 5.1 Load pre-trained model
For this image classification exercise, we leveraged 8 different pre-trained models as a part of transfer learning approach:

1. Resnet18, Resnet34, Resnet50 and Resnet101
2. VGG16 and VGG19
3. EfficientNet
4. Inception

All the models have been loaded from the torchvision module. (https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

By default, when we load a pretrained model all of the parameters have .requires_grad=True. But we trained only the last layer hence we set the attribute requires_grad=False.

"""

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

def initialize_model(device, model_name, num_classes, use_pretrained=True):
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    
    elif model_name == "vgg19":
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    
    elif model_name == "EfficientNet":
        from efficientnet_pytorch import EfficientNet
        model_ft = EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
            param.aux_logits = False    
        num_ftrs = model_ft.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        model_ft.aux_logits = False
    
    else:
        print("Invalid model name, exiting...")
        exit()
    
    model_ft.to(device)
    return model_ft
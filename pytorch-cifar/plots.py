import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
from models import *


# graph of model accuracies in function of number of parameters
accuracies = {"VGG16":92.64,
"ResNet18":93.02,
"ResNet50":93.62,
"ResNet101":93.75,
"RegNetX_200MF":94.24,
"RegNetY_400MF":94.29,
"MobileNetV2":94.43,
"ResNeXt29(32x4d)":94.73,
"ResNeXt29(2x64d)":94.82,
"SimpleDLA":94.89,
"DenseNet121":95.04,
"PreActResNet18":95.11,
"DPN92":95.16,
"DLA":95.47}   


# Function to count parameters for each model
def count_model_parameters():
    """Create a dictionary with parameter counts for each model"""
    
    model_configs = {
        'VGG16': VGG('VGG16'),
        'ResNet18': ResNet18(),
        'ResNet50': ResNet50(),
        'ResNet101': ResNet101(),
        'RegNetX_200MF': RegNetX_200MF(),
        'RegNetY_400MF': RegNetY_400MF(),
        'MobileNetV2': MobileNetV2(),
        'ResNeXt29(32x4d)': ResNeXt29_32x4d(),
        'ResNeXt29(2x64d)': ResNeXt29_2x64d(),
        'SimpleDLA': SimpleDLA(),
        'DenseNet121': DenseNet121(),
        'PreActResNet18': PreActResNet18(),
        'DPN92': DPN92(),
        'DLA': DLA(),
    }
    
    model_params = {}
    
    for model_name, model in model_configs.items():
        try:
            total_params = sum(p.numel() for p in model.parameters())
            model_params[model_name] = total_params
            print(f"{model_name:20s}: {total_params:,} parameters")
        except Exception as e:
            print(f"{model_name:20s}: Error - {e}")
    
    return model_params


# Generate parameter counts
if __name__ == "__main__":
    params_dict = count_model_parameters()
    
    # Print summary sorted by parameter count
    print("\n" + "="*50)
    print("Models sorted by parameter count:")
    print("="*50)
    for model_name in sorted(params_dict, key=params_dict.get):
        print(f"{model_name:20s}: {params_dict[model_name]:,}")

    plt.figure(figsize=(14, 8))
    plt.scatter(list(params_dict.values()), list(accuracies.values()), color='blue', s=100)
    plt.xscale('log')
    
    # Add model name labels for each point
    for model_name, acc in accuracies.items():
        params = params_dict[model_name]
        plt.annotate(model_name, xy=(params, acc), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
    #Add two models Resnet18 with data augmentation and Resnet18 with data augmentation and mixup
    plt.scatter(11_000_000, 95.54, color='red', s=100, label='ResNet18 + Augmentation')
    plt.scatter(11_000_000, 95.57, color='green', s=100, label='ResNet18 + Augmentation + Mixup')
    plt.legend(loc='lower right')
    
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Accuracy vs. Number of Parameters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_vs_parameters.png', dpi=150)
    plt.show()


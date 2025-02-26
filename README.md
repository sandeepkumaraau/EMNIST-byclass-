# EMNIST-byclass-
This project focuses on handwritten letter and character recognition using a pre-trained Convolutional Neural Network (CNN). The model is fine-tuned to achieve high accuracy on the given dataset. To enhance generalization and mitigate overfitting, various data augmentation techniques have been applied. 

I did this project in google colab pro and used T4 Gpu and i have written more about it , in the project pdf.

!pip install torcheval
!pip install torch_optimizer
!pip install torchmetrics
!pip install torchvision
!pip install lion-pytorch
!pip install timm

import torch
import cv2
import random
import numpy as np
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn



from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from torch.optim.optimizer import Optimizer
from lion_pytorch import Lion
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, SequentialLR

from torch.cuda.amp import autocast, GradScaler
from torchmetrics import F1Score
from timm.data.mixup import Mixup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HISTEq:
    def __call__(self, img):
      img_np = np.array(img)
      if img_np.dtype != np.uint8:
         img_np = (img_np * 255).astype(np.uint8)


      if len(img_np.shape) == 2:
        equalized = cv2.equalizeHist(img_np)
      else:
         raise ValueError("Expected single-channel image")

      return transforms.functional.to_pil_image(equalized)



import pandas as pd
import torch
import os
from skimage import io
from torch.utils.data import Dataset

class ImageNet(Dataset):

  def __init__(self, data_file, root_dir, transform = None):
    self.annotations = pd.read_csv("val.txt", delim_whitespace=True)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    image = io.imread(img_path)
    y_label = torch.Tensor(int(self.annotations.iloc[index, 1]))

    if self.transform:
      image = self.transform(image)
    
    return (image, y_label)
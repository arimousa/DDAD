import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import pydicom
from pydicom.pixel_data_handlers import apply_windowing












# import numpy as np
# import pynetdicom
# from PIL import ImageTk, Image
# import torch
# import cv2
# import glob
# from matplotlib import pyplot as plt
# from visualize import show_tensor_image

# dcm =  pynetdicom.dicom.read_file('462822612.dcm/462822612.dcm')
# img = dcm.pixel_array

# img = (img - img.min()) / (img.max() - img.min())
# img = torch.from_numpy(img)
# img = img.unsqueeze(0)

# plt.figure(figsize=(11,11))
# plt.axis('off')
# plt.subplot(1, 1, 1)
# plt.imshow(show_tensor_image(img))
# plt.title('kaggle')
# plt.savefig('Kaggle.png')
# plt.close()
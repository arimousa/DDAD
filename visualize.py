import matplotlib.pyplot as plt
from utilities import *


def visualize(image, noisy_image, GT,anomaly_map, index) :
    for idx, img in enumerate(image):
        plt.figure(figsize=(11,11))
        plt.axis('off')
        plt.subplot(1, 4, 1)
        plt.imshow(show_tensor_image(image[idx]))
        plt.title('clear image')


        plt.subplot(1, 4, 2)
        plt.imshow(show_tensor_image(noisy_image[idx]))
        plt.title('reconstructed image')
        

       
        plt.subplot(1, 4, 3)
        plt.imshow(show_tensor_image(GT[idx]))
        plt.title('ground truth')

        plt.subplot(1, 4, 4)
        plt.imshow(show_tensor_image(anomaly_map[idx]))
        plt.title('resutl')
        plt.savefig('results/sample{}.png'.format(index+idx))


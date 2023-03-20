
from utils import get_dataset, getGroundTruth, getImage
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
import matplotlib as mpl

IMAGE = 0
LABEL = 1
def display_images(batch):
    # TODO - Your implementation here
    mpl.use("macosx")
    fig, ax = plt.subplots(2,5,figsize = (20,10))
    
    for idx, data in enumerate(batch):
        if idx == 10:
            break
        # img = data["image"].numpy()
        
        img = getImage(data[IMAGE])
        ax[int(idx/5), idx%5].imshow(img)
        gtBoxes , gtClasses = getGroundTruth(data[LABEL])
        BoxAndClass = zip(gtBoxes, gtClasses)
        for box, cl in BoxAndClass:
            # box = box.numpy()*640
            # cl = tf.squeeze(cl)
            center_y, center_x = box["center_y"], box["center_x"]
            height, width  = box["length"], box["width"]
            x = center_x - height / 2
            y = center_y - width / 2
            if cl == 1:
                color = 'red'
            elif cl == 2:
                color = 'blue'
            elif cl == 4:
                color = 'green'
            # rect = patches.Rectangle((x1,y1),x2-x1,y2-y1, linewidth = 2, edgecolor  = color, fill= False)
            rect = patches.Rectangle((x, y), height,width, linewidth=2, edgecolor = color, facecolor='none')
            ax[int(idx/5), idx%5].add_patch(rect)
    fig.tight_layout(pad = 0.5)
    plt.show()
    
    
dataset = get_dataset("dataset/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord")

display_images(dataset)



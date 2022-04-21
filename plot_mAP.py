#! /usr/bin/python3

from matplotlib import legend
import numpy as np
import matplotlib.pyplot as plt

mAP = np.array(
    [
        [7.25, 8.32, 30.87, 31.33, 31.65, 34.31],
        [36.38, 35.11, 58.61, 57.75, 58.56, 58.40],
        [55.61, 53.3, 56.42, 53.35, 55.89, 54.37],
        [35.46, 35.8, 46.02, 44.65, 51.15, 50.52],
        [8.62, 8.83, 29.84, 30.86, 40.72, 41.37],
    ]
)


bbox_mAP_1 = mAP[:, 0]
mask_mAP_1 = mAP[:, 1]
bbox_mAP_2 = mAP[:, 2]
mask_mAP_2 = mAP[:, 3]
bbox_mAP_3 = mAP[:, 4]
mask_mAP_3 = mAP[:, 5]

x = np.array([1, 2, 3, 4, 5])
legends = ["RGBD w/o depth augmentation", "RGBD w/ depth augmentation", "RGB only"]
w = 0.2
margin = 0.05
plt.figure(figsize=(12, 9))
ax = plt.subplot(1, 1, 1)
ax.bar(x - 0.5 * (2 * w + margin), bbox_mAP_1, width=w, color="b", align="center")
ax.bar(x, bbox_mAP_2, width=w, color="g", align="center")
ax.bar(x + 0.5 * (2 * w + margin), bbox_mAP_3, width=w, color="r", align="center")
ax.autoscale(tight=True)
plt.title("Bounding Box mAP for 0.5 IoU threshold")
plt.xlabel("test dataset index")
plt.ylabel("mAP")
plt.legend(legends)
plt.axis((0, 6, 0, 60))
plt.show()

plt.figure(figsize=(12, 9))
ax = plt.subplot(1, 1, 1)
ax.bar(x - 0.5 * (2 * w + margin), mask_mAP_1, width=w, color="b", align="center")
ax.bar(x, mask_mAP_2, width=w, color="g", align="center")
ax.bar(x + 0.5 * (2 * w + margin), mask_mAP_3, width=w, color="r", align="center")
ax.autoscale(tight=False)
plt.title("Mask mAP for 0.5 IoU threshold")
plt.xlabel("test dataset index")
plt.ylabel("mAP")
plt.legend(legends)
plt.axis((0, 6, 0, 60))
plt.show()

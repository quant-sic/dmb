from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from dmb.paths import REPO_DATA_ROOT

if __name__ == "__main__":
    img = mpimg.imread(REPO_DATA_ROOT / "bose_hubbard_2d/minerva/minerva.png")

    fig, ax = plt.subplots()
    ax.imshow(img)

    # downsample img to 35x35
    img_downsampled = cv2.resize(img, (35, 35), interpolation=cv2.INTER_AREA)
    # to black and white
    img_downsampled = cv2.cvtColor(img_downsampled, cv2.COLOR_BGR2GRAY)

    minerva = (
        1
        - (img_downsampled - img_downsampled.min())
        / (img_downsampled.max() - img_downsampled.min())
    ) * 2.8

    # save minerva as .npy
    import numpy as np

    np.save(REPO_DATA_ROOT / "bose_hubbard_2d/minerva/minerva.npy", minerva)

    # checkerboard
    checkerboard = np.zeros((35, 35))
    checkerboard[::2, ::2] = 1
    checkerboard[1::2, 1::2] = 1

    minerva_half_checkerboard = minerva
    minerva_half_checkerboard[:, :17] = checkerboard[:, :17] * minerva[:, :17]

    np.save(
        REPO_DATA_ROOT / "bose_hubbard_2d/minerva/minerva_half_checkerboard.npy",
        minerva_half_checkerboard,
    )

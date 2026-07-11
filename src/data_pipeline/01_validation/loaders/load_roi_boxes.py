from scipy.io import loadmat
import numpy as np


def load_roi_boxes(mat_path):

    data = loadmat(mat_path)

    print("\n===== ROI FILE CONTENTS =====")

    for key in data.keys():
        if not key.startswith("__"):
            print(key)

    return data
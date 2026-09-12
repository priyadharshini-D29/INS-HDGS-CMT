import numpy as np


def fuse_features(eeg_features, et_features):
    """Concatenate EEG and ET feature vectors into one fusion vector."""
    return np.concatenate([eeg_features, et_features]).astype(np.float32)


def build_feature_matrix(eeg_feature_list, et_feature_list):
    """
    Stack per-epoch feature lists into (N, D_eeg), (N, D_et), (N, D_fusion).
    """
    eeg_mat    = np.stack(eeg_feature_list, axis=0)
    et_mat     = np.stack(et_feature_list,  axis=0)
    fusion_mat = np.concatenate([eeg_mat, et_mat], axis=1)
    return eeg_mat, et_mat, fusion_mat

import numpy as np


def validate_channel_correlation(eeg_stream):

    eeg = np.array(eeg_stream["time_series"])

    print("\n===== EEG CHANNEL CORRELATION =====")

    channel_std = np.std(eeg, axis=0)

    valid_channels = channel_std > 1e-6

    eeg_valid = eeg[:, valid_channels]

    corr = np.corrcoef(eeg_valid.T)

    mean_corr = np.mean(np.abs(corr))

    print(f"Mean absolute correlation : {mean_corr:.4f}")

    highly_corr = np.sum(np.abs(corr) > 0.95)

    print(f"Highly correlated pairs   : {highly_corr}")

    bad_channels = np.where(~valid_channels)[0]

    print(f"Excluded bad channels : {bad_channels}")

    return corr
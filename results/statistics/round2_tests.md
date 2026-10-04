# Round-2 tests from saved per-fold CSVs

## A. Engagement index (37 folds): paired differences, first minus second

balanced_acc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC) | 37 | -0.037 [-0.089, +0.011] | 0.278 | 1.000 | -0.25 | 12/13/12
  decoder vs EEGNet+ET-LSTM | 37 | -0.012 [-0.067, +0.038] | 0.914 | 1.000 | -0.02 | 15/8/14
  decoder vs ShallowConvNet+ET-LSTM | 37 | +0.001 [-0.075, +0.074] | 0.885 | 1.000 | +0.03 | 16/10/11
  ET-LSTM vs EEGNet+ET-LSTM | 37 | +0.025 [-0.032, +0.079] | 0.309 | 1.000 | +0.23 | 15/11/11
  ET-LSTM vs ShallowConvNet+ET-LSTM | 37 | +0.038 [-0.017, +0.092] | 0.142 | 0.712 | +0.32 | 17/10/10

roc_auc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC) | 37 | -0.011 [-0.048, +0.022] | 0.856 | 0.856 | -0.05 | 10/18/9
  decoder vs EEGNet+ET-LSTM | 37 | +0.067 [+0.010, +0.123] | 0.018 | 0.072 | +0.53 | 21/11/5
  decoder vs ShallowConvNet+ET-LSTM | 37 | +0.045 [-0.018, +0.116] | 0.241 | 0.483 | +0.28 | 14/14/9
  ET-LSTM vs EEGNet+ET-LSTM | 37 | +0.078 [+0.037, +0.124] | 0.001 | 0.007 | +0.73 | 19/12/6
  ET-LSTM vs ShallowConvNet+ET-LSTM | 37 | +0.056 [+0.010, +0.110] | 0.073 | 0.219 | +0.46 | 12/17/8

mcc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC) | 37 | -0.071 [-0.172, +0.022] | 0.277 | 0.830 | -0.26 | 10/13/14
  decoder vs EEGNet+ET-LSTM | 37 | -0.014 [-0.113, +0.081] | 0.938 | 1.000 | +0.02 | 17/6/14
  decoder vs ShallowConvNet+ET-LSTM | 37 | +0.021 [-0.126, +0.165] | 0.704 | 1.000 | +0.08 | 17/7/13
  ET-LSTM vs EEGNet+ET-LSTM | 37 | +0.058 [-0.049, +0.160] | 0.172 | 0.687 | +0.30 | 19/9/9
  ET-LSTM vs ShallowConvNet+ET-LSTM | 37 | +0.092 [-0.012, +0.201] | 0.069 | 0.344 | +0.39 | 20/8/9

## B. Fold-wise label minus pooled label (Table S17 rows)

balanced_acc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  EEGNet: fold-wise minus pooled | 37 | +0.051 [-0.014, +0.119] | 0.211 | 0.633 | +0.25 | 21/4/12
  ShallowConvNet: fold-wise minus pooled | 37 | +0.010 [-0.066, +0.086] | 0.802 | 1.000 | +0.05 | 17/4/16
  ET-LSTM: fold-wise minus pooled | 37 | -0.018 [-0.065, +0.024] | 0.668 | 1.000 | -0.10 | 13/13/11

roc_auc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  EEGNet: fold-wise minus pooled | 37 | +0.019 [-0.081, +0.117] | 0.626 | 1.000 | +0.10 | 18/3/16
  ShallowConvNet: fold-wise minus pooled | 37 | +0.026 [-0.053, +0.112] | 0.941 | 1.000 | -0.01 | 17/2/18
  ET-LSTM: fold-wise minus pooled | 37 | -0.031 [-0.057, -0.007] | 0.021 | 0.062 | -0.58 | 8/16/13

mcc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  EEGNet: fold-wise minus pooled | 37 | +0.096 [-0.032, +0.231] | 0.213 | 0.640 | +0.24 | 22/2/13
  ShallowConvNet: fold-wise minus pooled | 37 | +0.004 [-0.137, +0.150] | 0.900 | 1.000 | -0.02 | 16/1/20
  ET-LSTM: fold-wise minus pooled | 37 | -0.020 [-0.120, +0.072] | 0.802 | 1.000 | -0.05 | 12/9/16

## C. Purchase label (42 folds): decoder minus each standard model

balanced_acc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs EEGNet | 42 | +0.060 [+0.023, +0.097] | 0.003 | 0.003 | +0.52 | 28/0/14
  decoder vs ET-LSTM | 42 | +0.065 [+0.032, +0.099] | 0.001 | 0.001 | +0.58 | 31/0/11
  decoder vs EEGNet+ET-LSTM | 42 | +0.070 [+0.039, +0.102] | 0.000 | 0.000 | +0.66 | 31/0/11

roc_auc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs EEGNet | 42 | +0.122 [+0.074, +0.172] | 0.000 | 0.000 | +0.75 | 33/0/9
  decoder vs ET-LSTM | 42 | +0.087 [+0.044, +0.132] | 0.001 | 0.001 | +0.56 | 28/0/14
  decoder vs EEGNet+ET-LSTM | 42 | +0.125 [+0.074, +0.179] | 0.000 | 0.000 | +0.69 | 31/0/11

mcc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs EEGNet | 42 | +0.109 [+0.047, +0.172] | 0.001 | 0.002 | +0.56 | 30/0/12
  decoder vs ET-LSTM | 42 | +0.114 [+0.056, +0.173] | 0.001 | 0.002 | +0.57 | 31/0/11
  decoder vs EEGNet+ET-LSTM | 42 | +0.118 [+0.066, +0.170] | 0.000 | 0.000 | +0.68 | 33/0/9

## D. Purchase label: does an epoch model add anything beyond session-level dwell?
  skipped: src\data_pipeline\04_segmentation\output\engagement_product\product_epochs.csv missing

## E. Pooled calibration of the two new fusions (10 equal-width bins, 347 held-out epochs)
  (missing: results\baselines\dl_tuned\fold_probs\probs_eegnet_et.csv)
  (missing: results\baselines\dl_tuned\fold_probs\probs_shallow_et.csv)

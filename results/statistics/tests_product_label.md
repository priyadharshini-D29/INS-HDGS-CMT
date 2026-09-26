## 0. Run check (product-selection label)
  part files present: 14 | merged shallow: True | merged shallow_et: True
  Audited decoder: 42 folds  [OK]
  EEGNet: 42 folds  [OK]
  ShallowConvNet: 42 folds  [OK]
  ET-LSTM: 42 folds  [OK]
  EEGNet+ET-LSTM: 42 folds  [OK]
  ShallowConvNet+ET-LSTM: 42 folds  [OK]

## 1. Table 7 rows: mean +/- SD over folds
  model | folds | BalAcc | ROC-AUC | MCC
  Audited decoder | 42 | 0.566 +/- 0.091 | 0.606 +/- 0.118 | 0.112 +/- 0.151
  EEGNet | 42 | 0.506 +/- 0.099 | 0.485 +/- 0.120 | 0.003 +/- 0.162
  ShallowConvNet | 42 | 0.485 +/- 0.097 | 0.488 +/- 0.136 | -0.032 +/- 0.176
  ET-LSTM | 42 | 0.501 +/- 0.075 | 0.519 +/- 0.108 | -0.001 +/- 0.130
  EEGNet+ET-LSTM | 42 | 0.495 +/- 0.075 | 0.482 +/- 0.128 | -0.005 +/- 0.121
  ShallowConvNet+ET-LSTM | 42 | 0.502 +/- 0.080 | 0.524 +/- 0.125 | -0.002 +/- 0.146

## 2. Decoder minus each standard model (42 folds; Holm over five comparisons per metric)

balanced_acc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs EEGNet | 42 | +0.060 [+0.023, +0.097] | 0.0031 | 0.0031 | +0.52 | 28/0/14
  decoder vs ShallowConvNet | 42 | +0.081 [+0.047, +0.116] | 0.0001 | 0.0005 | +0.65 | 32/0/10
  decoder vs ET-LSTM | 42 | +0.065 [+0.032, +0.099] | 0.0007 | 0.0021 | +0.58 | 31/0/11
  decoder vs EEGNet+ET-LSTM | 42 | +0.070 [+0.039, +0.102] | 0.0001 | 0.0005 | +0.66 | 31/0/11
  decoder vs ShallowConvNet+ET-LSTM | 42 | +0.064 [+0.028, +0.099] | 0.0008 | 0.0021 | +0.58 | 30/0/12

roc_auc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs EEGNet | 42 | +0.122 [+0.074, +0.172] | 0.0000 | 0.0001 | +0.75 | 33/0/9
  decoder vs ShallowConvNet | 42 | +0.118 [+0.070, +0.170] | 0.0001 | 0.0002 | +0.68 | 32/0/10
  decoder vs ET-LSTM | 42 | +0.087 [+0.044, +0.132] | 0.0012 | 0.0024 | +0.56 | 28/0/14
  decoder vs EEGNet+ET-LSTM | 42 | +0.125 [+0.074, +0.179] | 0.0000 | 0.0001 | +0.69 | 31/0/11
  decoder vs ShallowConvNet+ET-LSTM | 42 | +0.082 [+0.031, +0.132] | 0.0045 | 0.0045 | +0.50 | 28/0/14

mcc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs EEGNet | 42 | +0.109 [+0.047, +0.172] | 0.0011 | 0.0025 | +0.56 | 30/0/12
  decoder vs ShallowConvNet | 42 | +0.144 [+0.082, +0.208] | 0.0001 | 0.0004 | +0.66 | 33/0/9
  decoder vs ET-LSTM | 42 | +0.114 [+0.056, +0.173] | 0.0009 | 0.0025 | +0.57 | 31/0/11
  decoder vs EEGNet+ET-LSTM | 42 | +0.118 [+0.066, +0.170] | 0.0001 | 0.0003 | +0.68 | 33/0/9
  decoder vs ShallowConvNet+ET-LSTM | 42 | +0.114 [+0.050, +0.179] | 0.0008 | 0.0025 | +0.57 | 31/0/11

## 3. ShallowConvNet minus EEGNet, alone and fused (Holm over two comparisons per metric)

balanced_acc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  ShallowConvNet vs EEGNet | 42 | -0.021 [-0.057, +0.015] | 0.2230 | 0.4459 | -0.22 | 17/0/25
  ShallowConvNet+ET-LSTM vs EEGNet+ET-LSTM | 42 | +0.007 [-0.024, +0.037] | 0.3585 | 0.4459 | +0.17 | 21/0/21

roc_auc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  ShallowConvNet vs EEGNet | 42 | +0.004 [-0.038, +0.046] | 0.9803 | 0.9803 | -0.01 | 20/0/22
  ShallowConvNet+ET-LSTM vs EEGNet+ET-LSTM | 42 | +0.043 [-0.003, +0.096] | 0.1346 | 0.2691 | +0.27 | 27/0/15

mcc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  ShallowConvNet vs EEGNet | 42 | -0.035 [-0.096, +0.026] | 0.2798 | 0.5597 | -0.19 | 18/0/24
  ShallowConvNet+ET-LSTM vs EEGNet+ET-LSTM | 42 | +0.004 [-0.053, +0.059] | 0.5114 | 0.5597 | +0.12 | 21/0/21

## 4. Incremental validity: does an epoch model add anything beyond session-level dwell?
  probe | mean per-fold ROC-AUC | pooled ROC-AUC | gain over dwell [95% CI], Wilcoxon p, W/T/L   (2338 epochs, 42 participants)
  log dwell alone | 0.880 | 0.854
  decoder probability alone | 0.606 | 0.591
  log dwell + decoder | 0.885 | 0.856 | +0.005 [-0.001, +0.012], p=0.106, 22/5/15
  eegnet probability alone | 0.485 | 0.510
  log dwell + eegnet | 0.879 | 0.853 | -0.002 [-0.003, -0.000], p=0.007, 6/20/16
  shallow probability alone | 0.487 | 0.494
  log dwell + shallow | 0.879 | 0.853 | -0.001 [-0.002, +0.000], p=0.175, 14/11/17
  et_lstm probability alone | 0.506 | 0.479
  log dwell + et_lstm | 0.879 | 0.852 | -0.002 [-0.004, -0.000], p=0.013, 8/18/16
  eegnet_et probability alone | 0.501 | 0.482
  log dwell + eegnet_et | 0.881 | 0.853 | +0.001 [-0.001, +0.003], p=0.765, 14/12/16
  shallow_et probability alone | 0.524 | 0.499
  log dwell + shallow_et | 0.880 | 0.853 | +0.000 [-0.001, +0.002], p=0.688, 18/11/13

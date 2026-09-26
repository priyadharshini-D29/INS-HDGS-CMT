# tests from saved per-fold CSVs
  (missing: results\baselines\dl_tuned\losocv_eegnet_et.csv)
  (missing: results\baselines\dl_tuned\losocv_shallow_et.csv)

## A. Engagement index (37 folds): paired differences, first minus second

balanced_acc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC) | 37 | -0.009 [-0.071, +0.051] | 0.770 | 0.770 | -0.07 | 14/15/8

roc_auc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC) | 37 | -0.011 [-0.048, +0.023] | 0.856 | 0.856 | -0.05 | 10/18/9

mcc: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L
  decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC) | 37 | -0.032 [-0.153, +0.085] | 0.615 | 0.615 | -0.12 | 12/15/10
  (missing: results\baselines\dl_tuned_v2\losocv_eegnet.csv)
  (missing: results\baselines\dl_tuned_v2\losocv_shallow.csv)
  (missing: results\baselines\dl_tuned_v2\losocv_et_lstm.csv)

## B. Fold-wise label minus pooled label (Table S17 rows)
  skipped: inputs missing
  (missing: results\label_product\baselines\dl_tuned\losocv_eegnet.csv)
  (missing: results\label_product\baselines\dl_tuned\losocv_et_lstm.csv)
  (missing: results\label_product\baselines\dl_tuned\losocv_eegnet_et.csv)

## C. Purchase label (42 folds): decoder minus each standard model
  skipped: inputs missing

## D. Purchase label: does an epoch model add anything beyond session-level dwell?
  probe | mean per-fold ROC-AUC | pooled ROC-AUC   (2338 epochs, 42 participants)
  log dwell alone | 0.880 | 0.854
  decoder probability alone | 0.606 | 0.591
  log dwell + decoder probability | 0.885 | 0.856   (gain over dwell alone: +0.005 per fold)
  (missing: results\label_product\baselines\dl_tuned\fold_probs\probs_eegnet.csv)
  (missing: results\label_product\baselines\dl_tuned\fold_probs\probs_et_lstm.csv)
  (missing: results\label_product\baselines\dl_tuned\fold_probs\probs_eegnet_et.csv)

## E. Pooled calibration of the two new fusions (10 equal-width bins, 347 held-out epochs)
  (missing: results\baselines\dl_tuned\fold_probs\probs_eegnet_et.csv)
  (missing: results\baselines\dl_tuned\fold_probs\probs_shallow_et.csv)

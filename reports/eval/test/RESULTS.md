# Results - test split (1041 images)
Model: `efficientnet_b0` @ 256px | operating threshold (chosen on val): **0.9995**

| metric (defective = positive) | tuned thr | thr 0.5 |
|---|---|---|
| precision | 1.0000 | 1.0000 |
| recall | 0.9850 | 0.9950 |
| F1 | 0.9925 | 0.9975 |
| accuracy | 0.9914 | 0.9971 |
| ROC-AUC | 1.0000 | |
| Avg. precision | 1.0000 | |

Confusion matrix (tuned): TN=439 FP=0 FN=9 TP=593

FN = missed defect (bad part ships). FP = false alarm (good part flagged).
Plots: confusion_matrix.png, pr_curve.png, roc_curve.png | Errors: error_montage_fp.png, error_montage_fn.png, errors/

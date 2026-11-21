| algoritmo | pré-processamento | F1-macro | dp | F1 anomaly | F1 normal | precisão anomaly | recall anomaly | acur. balanceada | acurácia | tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| HGB_native | redundância: majority_label; ausentes: NaN nativo; service: nativa | 0.9820 | 0.0015 | 0.9652 | 0.9988 | 0.9778 | 0.9531 | 0.9762 | 0.9977 | 16 |
| RF | redundância: none; ausentes: median; service: onehot; tree_select | 0.9813 | 0.0020 | 0.9638 | 0.9988 | 0.9708 | 0.9570 | 0.9780 | 0.9976 | 63 |
| XGB | redundância: majority_label; ausentes: NaN nativo; service: nativa | 0.9786 | 0.0013 | 0.9587 | 0.9986 | 0.9801 | 0.9382 | 0.9688 | 0.9973 | 18 |
| ET | redundância: none; ausentes: median; service: onehot; tree_select | 0.9741 | 0.0045 | 0.9499 | 0.9983 | 0.9667 | 0.9340 | 0.9664 | 0.9967 | 66 |
| MLP | redundância: none; ausentes: median; service: onehot; log1p + standard | 0.9641 | 0.0038 | 0.9305 | 0.9977 | 0.9534 | 0.9102 | 0.9543 | 0.9955 | 28 |
| KNN | redundância: none; ausentes: median; service: onehot; log1p + standard; kbest_f k=30 | 0.9518 | 0.0083 | 0.9068 | 0.9969 | 0.9322 | 0.8838 | 0.9408 | 0.9940 | 14 |
| DT | redundância: none; ausentes: median; service: onehot; tree_select | 0.8937 | 0.0045 | 0.7956 | 0.9917 | 0.6906 | 0.9389 | 0.9623 | 0.9841 | 35 |
| NB | redundância: none; ausentes: median; service: onehot; log1p + standard | 0.7763 | 0.0419 | 0.5768 | 0.9758 | 0.4282 | 0.9058 | 0.9308 | 0.9542 | 11 |
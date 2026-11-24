| modelo | pré-processamento | F1-macro | dp | F1 anomaly | F1 normal | precisão anomaly | recall anomaly | acur. balanceada | acurácia | tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Voto HGB+XGB+RF | cada membro com o seu | 0.8826 | 0.0029 | 0.7794 | 0.9859 | 0.7568 | 0.8035 | 0.8937 | 0.9734 | 663 |
| Voto HGB + LightGBM | cada membro com o seu | 0.8822 | 0.0030 | 0.7786 | 0.9859 | 0.7587 | 0.7996 | 0.8919 | 0.9734 | 138 |
| Voto HGB busca 2 ± atributos | cada membro com o seu | 0.8821 | 0.0027 | 0.7783 | 0.9859 | 0.7584 | 0.7993 | 0.8917 | 0.9734 | 117 |
| Voto HGB + LightGBM + CatBoost | cada membro com o seu | 0.8821 | 0.0029 | 0.7784 | 0.9859 | 0.7588 | 0.7991 | 0.8916 | 0.9734 | 276 |
| HGB busca 2, 5 sementes | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa | 0.8820 | 0.0027 | 0.7782 | 0.9859 | 0.7588 | 0.7987 | 0.8915 | 0.9734 | 224 |
| HGB busca 2 | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa | 0.8819 | 0.0030 | 0.7779 | 0.9858 | 0.7588 | 0.7982 | 0.8912 | 0.9734 | 76 |
| HGB busca 2 + atributos | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa; + atributos derivados | 0.8818 | 0.0030 | 0.7777 | 0.9858 | 0.7589 | 0.7976 | 0.8909 | 0.9734 | 77 |
| LightGBM | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa | 0.8815 | 0.0029 | 0.7772 | 0.9858 | 0.7589 | 0.7965 | 0.8904 | 0.9733 | 63 |
| Voto HGB + CatBoost | cada membro com o seu | 0.8813 | 0.0027 | 0.7767 | 0.9858 | 0.7594 | 0.7950 | 0.8897 | 0.9733 | 326 |
| RF | redundância: none; ausentes: median; service: onehot | 0.8812 | 0.0031 | 0.7768 | 0.9857 | 0.7551 | 0.7999 | 0.8919 | 0.9731 | 619 |
| HGB_native | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa | 0.8811 | 0.0030 | 0.7764 | 0.9857 | 0.7534 | 0.8011 | 0.8924 | 0.9731 | 71 |
| HGB + atributos | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa; + atributos derivados | 0.8803 | 0.0029 | 0.7749 | 0.9856 | 0.7552 | 0.7959 | 0.8900 | 0.9730 | 85 |
| XGB | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa | 0.8790 | 0.0039 | 0.7725 | 0.9855 | 0.7549 | 0.7910 | 0.8875 | 0.9728 | 68 |
| CatBoost | redundância: drop_inconsistent; ausentes: NaN nativo; service: nativa | 0.8770 | 0.0022 | 0.7686 | 0.9853 | 0.7531 | 0.7849 | 0.8845 | 0.9724 | 288 |
| MLP | redundância: none; ausentes: median; service: onehot; log1p + standard | 0.8716 | 0.0036 | 0.7587 | 0.9845 | 0.7362 | 0.7830 | 0.8828 | 0.9709 | 165 |
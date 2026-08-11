# Reference results

These are the values reported by the article’s final HGWO-BPSO configuration,
rounded as shown in the comparison tables.

| Dataset | Selected features | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| WUSTL-IIoT | 23 | 0.9999 | 0.9962 |
| Edge-IIoT | 27 | 0.9992 | 0.9919 |

The stored experiment outputs used for the article were approximately:

- WUSTL-IIoT: accuracy `0.999987`, macro-F1 `0.996158`.
- Edge-IIoT: accuracy `0.999205`, macro-F1 `0.991902`.

Small differences can occur across TensorFlow, CUDA, and GPU versions.

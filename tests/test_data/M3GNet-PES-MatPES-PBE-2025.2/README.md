---
library_name: matgl
tags:
- matgl
- materials-science
- graph-neural-network
- machine-learning-interatomic-potential
- foundation-potential
- mlip
---
# M3GNet-PES-MatPES-PBE-2025.2

## Introduction

Pre-trained M3GNet foundation potential, i.e., universal machine learning interatomic potential trained on the MatPES-PBE-2025.2 dataset.

## Potential

[matgl](https://github.com/materialyzeai/matgl) `Potential` model (version 3).

## Usage

```python
import matgl

model = matgl.load_model("materialyze/M3GNet-PES-MatPES-PBE-2025.2")
```

## Model Details

- Number of parameters: 288,157

## Metrics

| Split | Energy MAE (eV/atom) | Force MAE (eV/A) | Stress MAE (GPa) |
|---|---:|---:|---:|
| Train | 0.039029 | 0.158321 | 0.729370 |
| Validation | 0.042763 | 0.179450 | 0.870219 |
| Test | 0.043360 | 0.183160 | 0.868076 |

## Metadata

```json
{
  "dataset": "MatPES-PBE-2025.2",
}
```

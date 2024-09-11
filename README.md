# QATGBDT
Quantization-aware Training of Gradient Boosting Decision Trees

This repository contains the pretrained models in the research presented in "Efficient Integer-Only-Inference of Gradient Boosting Decision Trees on Low-Power Devices," focusing on quantization-aware training for optimizing GBDT models on FPGA platforms.

## Prerequisites
- Xilinx Vivado Design Suite
- Python 3
- Compatible FPGA hardware:(e.g. AMD Kria KV260)
- Conifer ([GitHub - thesps/conifer](https://github.com/thesps/conifer))
- CatBoost (Install via pip: `pip install catboost`)

## Installation
```bash
# Clone the repository
git clone https://github.com/malsharari/QATGBDT.git
# Navigate to the directory
cd QATGBDT
# Install dependencies and setup the project
```

## Binary Classification
Explore the binary classification examples to see how the model performs on tasks such as:
- [Network intrusion detection](Examples/Binary_classification_tasks/cybersecurity)

## Multiclass Classification
Explore the multiclass classification examples to see how the model performs on tasks such as:
- [Jet substructure classification](Examples/Multiclass_classification_tasks/jet_substructure/)

## Citation
Please cite our work as follows:
```
@article{10652593,
  author={Alsharari, Majed and Mai, Son T. and Woods, Roger and Reaño, Carlos},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
  title={Efficient Integer-Only-Inference of Gradient Boosting Decision Trees on Low-Power Devices},
  year={2024},
  pages={1-13},
  doi={10.1109/TCSI.2024.3446582}
}
```

## License
Distributed under the ASL License. See `LICENSE` for more information.
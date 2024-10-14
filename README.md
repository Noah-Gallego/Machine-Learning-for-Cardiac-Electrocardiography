# Machine Learning for Cardiac Electrophysiology 
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)
![Python](https://img.shields.io/badge/Python-3.8-blue.svg)

**If you find this repository useful, please consider citing the works listed in [Citations](#citations) and ⭐ this repository!**

This repository contains the code and notebooks from my **Data Science Challenge Project** at **Lawrence Livermore National Laboratory (LLNL)**, focused on **machine learning for cardiac electrophysiology**. This project was part of the challenge where participants used **machine learning techniques** to reconstruct electroanatomical maps from ECG signals.

My contributions expanded upon the initial framework by implementing advanced machine learning models for ECG signal analysis, including **sequence-to-sequence prediction** for transmembrane potentials and **activation map reconstruction** using **neural networks**. I utilized cutting-edge tools such as **PyTorch** to enhance the models’ performance and created new pipelines for data pre-processing and model evaluation.

[![Description](https://github.com/landajuela/cardiac_challenge/blob/main/figures/rotating_hearts.gif)](https://github.com/landajuela/cardiac_challenge/blob/main/figures/rotating_hearts.gif)

In particular, this repository includes:
- **Custom neural network architectures** for cardiac data.
- **Optimization techniques** tailored for high-dimensional ECG signals.
- An implementation of **Grad-CAM** to visualize the important regions of ECG signals that influence the network's decision-making process.

The repository also provides helpful functions to load and manipulate the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106), in addition to tutorials and notebooks for new users.

## Description

This project explores how machine learning can transform **standard 12-lead electrocardiograms (ECG)** into detailed maps of the heart’s electrical activity. Such data-driven approaches aim to replace expensive, invasive procedures with non-invasive ECG signals, enabling the identification of cardiac conditions like arrhythmias with greater precision. 

The challenge presented by LLNL was to:
1. Classify heartbeats from ECG data.
2. Reconstruct cardiac activation maps from simulated intracardiac voltage readings.
3. Use **sequence-to-sequence models** to predict the transmembrane potential across time.

## My Contributions
### Major Contributions:
- Developed **custom neural networks** for **activation map reconstruction** and **transmembrane potential prediction**, transforming a 12x500 ECG sequence into spatially relevant activation maps.
- Applied **deep learning models** using **PyTorch**, fine-tuning for improved performance on complex cardiac datasets.
- Integrated **Grad-CAM** to visualize how ECG signal segments influence model outputs, providing interpretability to the deep learning models.
- Automated the data preprocessing pipeline for large datasets, optimizing training workflows.

### Additional Enhancements:
- Enhanced code modularity and documentation for easier usability.
- Introduced **early stopping and learning rate schedulers** to improve model performance.
- Provided extensive tutorials for those new to machine learning in cardiology.

## Getting Started

For new users, this repository includes tutorials and notebooks to help you get started with the challenge. You can use **any machine learning framework** like **TensorFlow** or **PyTorch** to experiment with the tasks. The four key tasks are outlined below:

### Task 1: Heartbeat Classification
Get familiar with ECG data using the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat). The goal is to perform binary classification for **healthy** vs. **irregular** heartbeats.

Start by reading the notebook: [task_1_getting_started.ipynb](./task1ArrhythmiaClassifier.ipynb)

### Task 2: Irregular Heartbeat Classification
This task involves performing **multiclass classification** to diagnose irregular heartbeats from the same dataset.

Start by reading the notebook: [task_2_getting_started.ipynb](./task2ArrhythmiaMultiClassifier.ipynb)

### Task 3: Activation Map Reconstruction from ECG
For this task, I developed **sequence-to-vector prediction models** to reconstruct activation maps using the [Simulated Intracardiac Voltage Recordings and ECG Signals Dataset](https://library.ucsd.edu/dc/object/bb29449106). This task involves transforming an ECG sequence (12x500) into a spatial activation map (75x1).

Start by reading the notebook: [task_3_getting_started.ipynb](./task3ActivationTimePredictor.ipynb)

### Task 4: Transmembrane Potential Reconstruction from ECG
The most complex task focuses on **sequence-to-sequence** prediction using the same dataset. The goal is to predict transmembrane potentials over time (12x500 → 75x500) using deep neural networks.

Start by reading the notebook: [task_4_getting_started.ipynb](./taskFourCompleteNoah.ipynb)

## Resources
### Datasets and Notebooks:
- **Dataset**: [Simulated Intracardiac Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106)
- **Tutorials**:
  - [ECG Image Classification](./tutorials/image_classifier_tutorial_v1.2.ipynb)
  - [Regression Tutorial](./tutorials/DSC_regression-tutorial.ipynb)
- **Notebooks**: [task_1_getting_started](./notebooks/task_1_getting_started.ipynb), [task_2_getting_started](./notebooks/task_2_getting_started.ipynb), [task_3_getting_started](./notebooks/task_3_getting_started.ipynb), [task_4_getting_started](./notebooks/task_4_getting_started.ipynb)

## Citations
If you use this repository or the included datasets, please cite the following works:

```
@INPROCEEDINGS{10081783,
  author={Landajuela, Mikel and Anirudh, Rushil and Loscazo, Joe and Blake, Robert},
  title={Intracardiac Electrical Imaging Using the 12-Lead ECG: A Machine Learning Approach Using Synthetic Data},
  year={2022}
}
```
```
Landajuela, Mikel; Anirudh, Rushil; Blake, Robert (2022).
Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals.
```

## License
This repository is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

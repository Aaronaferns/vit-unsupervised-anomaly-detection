# Vision Transformer with Capsule Network for Industrial Anomaly Detection (VTIAD-Capsule)

## Overview
The **Vision Transformer-based Anomaly Detection** (VTIAD) network is enhanced with a **Capsule Network** to effectively detect and localize anomalies in industrial images. By leveraging the spatial preservation capabilities of Vision Transformers (ViT) and the dynamic routing of Capsule Networks, this approach aims to improve the accuracy and robustness of anomaly detection tasks.

The model consists of two main components:
1. **Vision Transformer**: This model divides the input image into patches, processes them through self-attention layers, and generates a representation that retains spatial information.
2. **Capsule Network**: The output of the Vision Transformer is passed through a Capsule Network to further enhance the detection and localization of anomalous regions. Capsule Networks capture part-whole relationships and improve the model’s ability to generalize across various types of anomalies.

The dataset used for training the model is **BeanTech Anomaly Detection Dataset (BTAD)**, specifically curated for industrial anomaly detection tasks.

## Model Details

- **Vision Transformer Architecture**: The input image is split into patches, and a transformer network processes these patches to learn global dependencies and spatial relationships.
  
- **Capsule Network**: The output embeddings from the Vision Transformer are passed through the Capsule Network, which uses dynamic routing to classify and localize anomalies by learning the spatial relationships between image features.

- **Gaussian Mixture Density Network (MDN)**: The final outputs are processed by an MDN to improve the localization and detection performance.

- **Regularization**: Gaussian noise is added to the encoded features for regularization, improving the model’s generalization ability.

## Dataset: BeanTech Anomaly Detection (BTAD)

The **BTAD dataset** contains high-resolution industrial images of products captured from industrial imaging systems. It includes three different products with varying image resolutions. Ground truth annotations are provided for precise anomaly localization.

- **Product 1**: 400 images of 1600x1600 pixels
- **Product 2**: 1000 images of 600x600 pixels
- **Product 3**: 399 images of 800x600 pixels

Images are processed with log transformations and privacy-preserving techniques, with pixel-precise ground truth annotations added.

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/vtiad-capsule.git
    cd vtiad-capsule
    ```

2. **Install dependencies**:
    Install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Dataset**:
    Download the BTAD dataset from [BeanTech's website](https://beantech.com/dataset) and place it in the `data` directory.

## Training

To train the model, run the following command:

```bash
python train.py --dataset "BTAD" --product "Product1" --epochs 100 --batch_size 32
```

### Arguments:
- `--dataset`: Specify the dataset. Currently supports "BTAD".
- `--product`: Choose from "Product1", "Product2", or "Product3".
- `--epochs`: Number of epochs for training.
- `--batch_size`: The batch size for training.

## Results

The model is evaluated using various anomaly detection metrics. The performance is benchmarked on both the BTAD dataset and publicly available datasets like MVTec. Results will be made available upon further testing.

## Cited

```
@inproceedings{Mishra2021VTAD,
  title={VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization},
  author={Pankaj Mishra and Ricardo Verk and Daniele Fornasier and Claudio Piciarelli and Gian Luca Foresti},
  booktitle={IEEE/IES International Symposium on Industrial Electronics (ISIE)},
  year={2021},
  publisher={IEEE}
}
```


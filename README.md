# 🫁 Lung Cancer Detection using CNN + Transfer Learning

A deep learning project for classifying lung histopathology images into
three categories using a custom baseline CNN and three pretrained models
(EfficientNetB0, InceptionV3, ResNet50) with Grad-CAM explainability.

**Author:** Ahmed Mohamed Abdelhady
**Dataset:** [LC25000 Lung and Colon Histopathological Image Dataset](https://arxiv.org/abs/1912.12378)

---

## Results

| Model | Test Accuracy | F1 (macro) | Mean AUC |
|---|---|---|---|
| Baseline CNN | 96.93% | 96.93% | 0.9973 |
| EfficientNetB0 | 99.47% | 99.47% | 0.9999 |
| InceptionV3 | 99.38% | 99.38% | 0.9999 |
| **ResNet50** | **99.82%** | **99.82%** | **1.0000** |

ResNet50 achieves perfect AUC (1.0000) and misses only 4 cancer cases
out of 1,500 test images across both cancer classes.

---

## Classes

| Class | Folder | Description |
|---|---|---|
| Lung Adenocarcinoma | `lung_aca` | Malignant — glandular tissue cancer |
| Normal | `lung_n` | Healthy lung tissue |
| Squamous Cell Carcinoma | `lung_scc` | Malignant — squamous cell cancer |

---

## Project Structure
```
lung-cancer-detection/
│
├── data/                          ← Not included in repo (too large)
│   └── lung_image_sets/
│       ├── lung_aca/              (5,000 images)
│       ├── lung_n/                (5,000 images)
│       └── lung_scc/             (5,000 images)
│
├── notebooks/
│   ├── 01_EDA.ipynb              ← Exploratory data analysis
│   ├── 02_preprocessing_check.ipynb
│   ├── 03_baseline_cnn.ipynb     ← Baseline CNN training
│   ├── 04_transfer_learning.ipynb ← Transfer learning training
│   ├── 05_evaluation.ipynb       ← Full model evaluation
│   └── 06_gradcam.ipynb          ← Grad-CAM visualizations
│
├── src/
│   ├── data_preprocessing.py     ← Data loading, splitting, augmentation
│   ├── model_baseline.py         ← Custom CNN architecture
│   ├── model_efficientnet.py     ← EfficientNetB0 architecture
│   ├── model_inception.py        ← InceptionV3 architecture
│   ├── model_resnet50.py         ← ResNet50 architecture
│   ├── model_loader.py           ← Centralized model loading
│   ├── train.py                  ← Standalone training script
│   ├── evaluate.py               ← Standalone evaluation script
│   └── predict.py                ← Single image prediction script
│
├── saved_models/                  ← Not included in repo (too large)
│   ├── baseline_cnn_weights.npy
│   ├── efficientnetb0_weights.npy
│   ├── inceptionv3_weights.npy
│   └── resnet50_weights.npy
│
├── results/                       ← All plots and evaluation outputs
│   ├── eda_class_distribution.png
│   ├── eda_sample_images.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── final_test_accuracy_comparison.png
│   ├── gradcam_examples.png
│   └── gradcam_average_per_class.png
│
├── ui/
│   └── app.py                    ← Gradio web interface
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/ahmedmohamedabdelhadi/lung-cancer-detection.git
cd lung-cancer-detection
```

### 2. Create the conda environment
```bash
conda create -n lung_cancer python=3.9.18
conda activate lung_cancer
conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge
pip install -r requirements.txt
```

### 3. Download the dataset
Download the [LC25000 dataset](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)
and place it at `data/lung_image_sets/` so the folder structure matches above.

---

## Usage

### Train all models
```bash
python src/train.py
python src/train.py --data_dir data/lung_image_sets --epochs 30
```

### Evaluate all models on the test set
```bash
python src/evaluate.py
```

### Predict a single image
```bash
python src/predict.py --image path/to/image.jpg
python src/predict.py --image path/to/image.jpg --model efficientnetb0
```

### Launch the Gradio web UI
```bash
python ui/app.py
```
Then open `http://localhost:7860` in your browser.

---

## Model Architecture

### Baseline CNN
A custom CNN trained from scratch as the benchmark model:
- 4 × Convolutional Blocks (Conv2D → BatchNorm → ReLU → MaxPool)
- Filters: 32 → 64 → 128 → 256
- GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(3, softmax)

### Transfer Learning Models
All three pretrained models follow the same two-stage strategy:

**Stage 1 — Feature extraction (5 epochs)**
The pretrained base is frozen. Only the classification head is trained.

**Stage 2 — Fine-tuning (up to 25 epochs)**
The last 30 layers of the base are unfrozen and trained at lr=1e-5.

Classification head (same for all three):
GlobalAveragePooling2D → Dropout(0.3) → Dense(3, softmax)

---

## Explainability — Grad-CAM

Grad-CAM visualizations confirm the model focuses on actual tissue
pathology rather than image artifacts:

- **Adenocarcinoma** — attention on irregular glandular cell clusters
- **Normal** — attention on organized structural tissue features
- **Squamous Cell Carcinoma** — tight focus on keratinizing cell nests

Each class produces a visually distinct attention pattern, confirming
the model has learned genuine class-specific tissue morphology.

---

## Key Technical Decisions

**Why numpy for weight saving?**
TF 2.10 has a serialization bug where `ReduceLROnPlateau` converts the
learning rate to an EagerTensor that cannot be JSON-serialized during
`model.save()`. Using `model.get_weights()` → `np.save()` bypasses
all format issues completely.

**Why preprocessing in the generator, not the model?**
Lambda layers inside Keras models cause weight index mismatches during
`load_weights()` in TF 2.10. Moving preprocessing to `ImageDataGenerator`
via `preprocessing_function` is fully compatible with all save/load operations.

**Why GlobalAveragePooling instead of Flatten?**
Flatten on a 14×14×256 feature map gives 50,176 values. GlobalAveragePooling
compresses to 256 values — 200× smaller, faster, and better regularized.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.9.18 |
| TensorFlow | 2.10.0 |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| NumPy | 1.23.5 |
| Gradio | 3.50.2 |
| GPU | NVIDIA GTX 1650 Ti |

---



---

## Contact

**Ahmed Mohamed Abdelhady**

- 📧 Email: [ahmed.mohamed.abdelhady01@gmail.com](mailto:ahmed.mohamed.abdelhady01@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/ahmed-mohamed-abdelhady](https://www.linkedin.com/in/ahmed-mohamed-abdelhady/)

Feel free to reach out for questions, collaboration, or feedback on this project.


## Disclaimer

This project is for **research and educational purposes only**.
It is not intended for clinical use and should not be used as a
substitute for diagnosis by a qualified medical professional.

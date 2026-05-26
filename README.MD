# Image Captioning using GANs Architecture

## Project Summary

This homework project implements a simple image captioning system using a GAN-style architecture on the COCO Captioning dataset.

The notebook file is:

```text
Image_Captioning_GAN.ipynb
```

The goal is to generate a short natural-language caption for an input image. The project uses a small subset of the COCO validation images so the training process is easier to understand, run, and explain.

Dataset source:

```text
https://cocodataset.org/#captions-2015
```

## Main Idea

The model has two main parts:

1. **Generator**
   - Uses MobileNetV2 as a CNN image encoder.
   - Uses an LSTM decoder to generate captions word by word.
   - Learns to produce captions from image features.

2. **Discriminator**
   - Receives image features and a caption.
   - Predicts whether the caption is a real COCO caption or a generated caption.
   - Helps the generator make captions that look more realistic.

The generator is trained first using normal supervised caption loss. After that, a simple GAN fine-tuning step is added using adversarial loss.

## Notebook Workflow

The notebook performs these steps:

1. Install and import required Python libraries.
2. Download COCO caption annotations.
3. Download a small number of COCO validation images.
4. Preprocess captions and build a vocabulary.
5. Create a PyTorch dataset and dataloader.
6. Build the CNN encoder, LSTM generator, and discriminator.
7. Train the generator using supervised learning.
8. Fine-tune using a simple GAN-style training loop.
9. Evaluate generated captions using BLEU scores.
10. Save evaluation results and model checkpoint.

## Output Files

After running all cells in the notebook, the following files are generated:

```text
results/metrics.json
results/evaluation_results.csv
results/image_captioning_gan_small.pt
```

File descriptions:

- `metrics.json`: stores BLEU scores and dataset summary.
- `evaluation_results.csv`: stores generated captions and reference captions.
- `image_captioning_gan_small.pt`: stores the trained model checkpoint.

## Current Evaluation Results

The current run used:

```text
Training images: 240
Validation images: 60
Vocabulary size: 832
```

Evaluation metrics:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.3746 |
| BLEU-2 | 0.1969 |
| BLEU-4 | 0.0391 |

These scores are expected for a small demonstration model trained on only a small subset of COCO. The purpose of this homework is to show the architecture and workflow clearly, not to achieve state-of-the-art captioning performance.

## Example Prediction

Example row from `results/evaluation_results.csv`:

```text
Image: COCO_val2014_000000397278.jpg
Predicted caption: a man is on a <unk>
Reference caption: A yellow sign sitting on the side of a road.
```

Because the dataset subset and training time are small, some generated captions may be generic or contain `<unk>` for unknown words.

## How to Run

Open and run all cells in:

```text
Image_Captioning_GAN.ipynb
```

Recommended environment:

```text
Google Colab with GPU enabled
```

or a local Python environment with:

```text
torch
torchvision
pandas
matplotlib
tqdm
pillow
```

## Short Explanation for Lecturer

This project demonstrates image captioning with a conditional GAN idea. The generator creates captions from image features, while the discriminator checks whether a caption looks real for the given image. The final model is evaluated using BLEU scores and saved prediction results.

The implementation is intentionally simple so the training process, model architecture, and evaluation method can be explained clearly.

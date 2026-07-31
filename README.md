# Momus

## Robust Document Forgery Detection and Localization

Momus is a research-oriented computer vision project for detecting and
localizing forged regions in identity documents. The project formulates
document forgery analysis as a pixel-wise segmentation task and provides
an end-to-end pipeline covering dataset preparation, synthetic data
generation, forgery simulation, model training, evaluation, robustness
analysis, and deployment.

The system is designed for KYC onboarding, financial services,
insurance, and document verification workflows where forged identity
documents can introduce significant operational and security risks.

------------------------------------------------------------------------

## Features

-   Pixel-level forgery localization
-   Dual-stream RGB + SRM segmentation architecture
-   Synthetic document generation for offline experimentation
-   Multiple forgery simulation strategies
-   Robustness evaluation under realistic image degradations
-   Cross-manipulation generalization experiments
-   ONNX export and inference benchmarking
-   Automated evaluation and reporting

------------------------------------------------------------------------

## Problem Statement

Modern identity verification systems must identify manipulated documents
before downstream processing. Typical attacks include copy-move forgery,
region splicing, text substitution, field replacement, and object
removal through inpainting.

Unlike binary document classifiers, Momus predicts a dense tampering
mask, enabling reviewers to identify both whether a document has been
manipulated and where the manipulation occurred.

------------------------------------------------------------------------

## Dataset

The project supports MIDV-500, MIDV-2020, and SROIE datasets. When
public datasets are unavailable, the repository automatically falls back
to a synthetic document generation pipeline to validate the complete
workflow. Synthetic data is intended only for pipeline verification and
not for reporting benchmark performance.

------------------------------------------------------------------------

## Architecture

The default implementation employs a dual-stream encoder-decoder
architecture:

-   RGB encoder for semantic appearance features
-   SRM residual encoder for forensic noise cues
-   Feature fusion at the bottleneck
-   Decoder with skip connections
-   Dice + BCE optimization objective

An optional SegFormer backbone is also supported for experimentation.

------------------------------------------------------------------------

## Training Pipeline

1.  Dataset preparation
2.  Synthetic fallback (optional)
3.  Forgery generation
4.  Image degradation
5.  Dataset construction
6.  Model training
7.  Evaluation
8.  Robustness testing
9.  Deployment

------------------------------------------------------------------------

## Evaluation

The evaluation framework includes:

-   Pixel IoU
-   Pixel F1
-   Image-level AUC
-   Cross-manipulation generalization
-   Robustness analysis
-   Deployment benchmarking

Metrics generated using synthetic data should be interpreted as pipeline
validation rather than representative production performance.

------------------------------------------------------------------------

## Repository Structure

``` text
momus/
├── config.yaml
├── requirements.txt
├── run_all.sh
├── data/
├── model/
├── eval/
├── deploy/
├── checkpoints/
├── results/
└── utils/
```

------------------------------------------------------------------------

## Installation

``` bash
git clone <repository-url>
cd momus
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

``` bash
./run_all.sh
```

or

``` bash
python data/download.py
python data/forge.py
python data/dataset.py
python model/train.py
python eval/evaluate.py
python eval/cross_manipulation.py
python eval/robustness.py
python deploy/export_onnx.py
python deploy/quantize.py
python deploy/benchmark.py
```

------------------------------------------------------------------------

## Configuration

All dataset paths, model settings, optimization parameters, and
experiment configurations are managed through `config.yaml`.

------------------------------------------------------------------------

## Limitations

-   Synthetic data is intended only for validating the software
    pipeline.
-   Performance should be evaluated using public benchmark datasets.
-   Deployment benchmarks depend on the target hardware platform.

------------------------------------------------------------------------

## Future Work

-   Transformer-based segmentation backbones
-   Multilingual document support
-   Additional forgery generation strategies
-   Confidence estimation and uncertainty calibration
-   Expanded benchmark evaluation

------------------------------------------------------------------------

## License

This project is intended for research and educational purposes.

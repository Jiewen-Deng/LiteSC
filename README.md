# LiteSC: A Lightweight Semantic Communication System with Hessian-Aware Pruning and Nonuniform-to-Uniform Quantization

Emerging edge-intelligence applications, such as virtual/augmented reality and the Internet of Things, require the reliable delivery of high-volume data under stringent latency and resource constraints. Semantic communication (SC), which transmits task-relevant meaning rather than raw bits, offers a promising route to improve bandwidth efficiency. However, existing SC transceivers rely heavily on deep neural networks (DNNs), incurring substantial computation and memory overhead that hinders deployment on resource-constrained edge devices. In addition, continuous-valued semantic features are not directly compatible with conventional digital communication hardware that operates on discrete constellations. To address these challenges, we propose LiteSC, a lightweight and digital-friendly SC system that jointly reduces computational and communication overhead while enabling practical digital modulation. We first introduce a SC framework with a learnable nonuniform-to-uniform quantization (N2UQ) module to convert continuous semantic representations into uniformly spaced discrete levels suitable for constellation mapping. Then, we develop a Hessian-pruned transceiver by removing redundant neurons based on second-order loss sensitivity, substantially reducing parameters and FLOPs. Finally, we apply N2UQ to both the pruned model parameters and the semantic features, achieving further compression with minimal fidelity loss. Experimental result demonstrate that LiteSC maintains high performance while drastically reducing resource requirements. Under Rician fading channels, LiteSC preserves 98.83\% BLEU score and 97.90\% semantic similarity score of the state-of-the-art DeepSC model, using only 6.26\% of the model size. Under AWGN channels, LiteSC outperforms magnitude-based pruning and uniform quantization baselines by 7.67\% and 23.58\% in BLEU score, respectively.

## Installation


   ```bash
   git clone https://github.com/Jiewen-Deng/LiteSC
   pip install -r requirements.txt
   ```

## Quick Start

1. Pretraining the quantization-based digital semantic communication system:
   ```bash
   python main_pretrain.py
   ```

   \# Evaluate the system with BLEU score:
   ```bash
   python performance_pretrain.py
   ```
   
2. Perform Hessian-aware pruning on pretrained digital semantic system:
   ```bash
   python main_HAP.py
   ```
   \# Evaluate the system with BLEU score:
   ```bash
   python performance_HAP.py
   ```

3. Perform nonuniform-to-uniform quantization on HAPed semantic system and evaluate the system with BLEU score and semantic similarity score:
   ```bash
   python main_N2UQ.py
   ```
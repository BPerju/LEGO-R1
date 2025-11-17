# LEGO-R1: Fine-grained VLMs Within Reinforcement Learning Frameworks

Bachelor thesis investigating the application of Group Relative Policy Optimization (GRPO) to spatial reasoning tasks in Vision-Language Models, specifically focused on the LEGO-ARTA assembly dataset.

## 📄 Thesis

**Fine-grained VLMs Within Reinforcement Learning Frameworks**  
*Bogdan Perju | Vrije Universiteit Amsterdam | 2025*

[Read Full Thesis](./Thesis.pdf)

### Abstract

This research investigates applying GRPO to spatial tasks within the LEGO-ARTA dataset, comparing three models: a few-shot baseline (Qwen2.5VL-7B-Instruct), an SFT-trained reference model, and a GRPO-optimized model. Custom reward functions target reasoning format, spatial understanding, and object detection accuracy.

**Key Results:**

- GRPO achieved **7.7% IoU improvement** over baseline (65.49 vs 60.82)
- Demonstrated superior spatial localization compared to SFT
- Revealed critical trade-offs between spatial accuracy and textual precision
- Provided insights into reward function design for RL-based VLM training

## 📊 Reproducibility & Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (A100 recommended for training, A40 for inference)
- 24GB+ VRAM for training

### Installation

**1. Clone VLM-R1 Project**
```bash
git clone https://github.com/om-ai-lab/VLM-R1
cd VLM-R1
```

**2. Add GRPO Training Script**

Copy `run_grpo_main.sh` from this repository to the VLM-R1 base directory.

**3. Replace Qwen Module**
```bash
# Navigate to Qwen module
cd src/open-r1-multimodal/src/open_r1/vlm_modules/

# Replace with custom module from this repository
cp iteration2/qwen_module.py qwen_module.py
```

**4. Prepare LEGO-ARTA Dataset**
```bash
# Download LEGO-ARTA using the provided notebook
# Then copy generated data
cp -r qwen_data/* /ARTA_LEGO/
```

### Training

**GRPO Training:**
```bash
bash run_grpo_main.sh
```

**SFT Training (via LLaMA-Factory):**

Use the generated data from `Qwen2.5-VL-DATA_WRANGLING.ipynb` - it's interchangeable between SFT and GRPO training.

**Iteration 1:** Runs inference using the provided notebook  
**Iteration 2:** Uses LLaMA-Factory for both SFT training and inference

### Technical Configuration

**Training:**

- GPU: Single A100
- LoRA rank: 8
- Precision: bf16
- Training steps: 500
- Optimization: DeepSpeed ZeRO Stage 2
- Vision backbone: Frozen

**Inference:**

- GPU: Single A40
- Batch size: 3
- Max pixels: 2600
- Flash-Attention-2 enabled

## 📈 Results

| Model | IoU ↑ | F1-Object ↑ | F1-State ↑ | FPR ↓ |
|-------|-------|------------|-----------|--------|
| Qwen2.5VL (Base) | 60.82 | 59.35 | 38.55 | 30.19 |
| Qwen2.5VL (SFT) | 14.16 | 81.14 | 100.00 | 0.00 |
| **Qwen2.5VL (GRPO)** | **65.49** | 62.96 | 39.66 | 36.18 |

## 🔗 Resources

- [LEGO-ARTA Dataset](https://arxiv.org/abs/2507.05515)
- [VLM-R1 Project](https://github.com/om-ai-lab/VLM-R1)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

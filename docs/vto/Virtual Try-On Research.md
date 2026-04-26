# Virtual Try-On Arxiv Papers Late 2025 - Early 2026

Research survey of VTO papers from late 2025 and early 2026.

## DiT-Based / Diffusion Transformer Approaches

### DiT-VTON
- **Paper**: [2510.04797](https://arxiv.org/abs/2510.04797)
- **Architecture**: Diffusion Transformer (DiT) framework adapted for VTO
- **Strategies explored**:
  - In-context token concatenation
  - Channel concatenation
  - ControlNet integration
- **Key features**: Multi-category Virtual Try-All (VTA) with integrated image editing, pose preservation, texture transfer

### TED-VITON
- **Paper**: [2411.17017](https://arxiv.org/abs/2411.17017)
- **Architecture**: DiT-based T2I backbone with Garment Semantic (GS) Adapter
- **Key innovations**:
  - Garment Semantic Adapter for enhancing garment-specific features
  - Text Preservation Loss for distortion-free text rendering on garments
  - LLM prompt optimization for constraint mechanisms
- **Date**: Submitted Nov 2024, latest version Mar 2025

### PROMO
- **Paper**: [2603.11675](https://arxiv.org/abs/2603.11675)
- **Architecture**: Flow Matching DiT backbone with latent multi-modal conditional concatenation
- **Key innovations**:
  - Self-reference mechanisms for inference acceleration
  - Structured image editing perspective
  - Efficient conditional generation

### TEMU-VTOFF (Inverse VTO)
- **Paper**: [2505.21062](https://arxiv.org/abs/2505.21062)
- **Architecture**: Dual DiT-based backbone with multimodal attention
- **Task**: Virtual Try-Off - generating flat product images from clothed individuals
- **Key innovations**: Text-enhanced framework, alignment module for detail preservation

## Training-Free / Zero-Shot Approaches

### OmniVTON
- **Paper**: [2507.15037](https://arxiv.org/abs/2507.15037)
- **Strategy**: Training-free universal VTON
- **Key components**:
  - Garment prior generation with alignment + boundary stitching
  - DDIM inversion for pose alignment
  - Decoupled garment and pose conditioning
- **Capabilities**: Multi-human VTON

### OmniVTON++
- **Paper**: [2602.14552](https://arxiv.org/abs/2602.14552)
- **Strategy**: Training-free with Principal Pose Guidance
- **Key components**:
  - Structured Garment Morphing for correspondence-driven adaptation
  - Principal Pose Guidance for step-wise structural regulation
  - Continuous Boundary Stitching for boundary refinement
- **Extensions**: Multi-garment, multi-human, anime character VTO

### DEFT-VTON
- **Paper**: [2509.13506](https://arxiv.org/abs/2509.13506)
- **Strategy**: Doob's h-transform efficient fine-tuning (1.42% trainable params)
- **Key innovations**:
  - Adaptive consistency loss combining distillation and denoising
  - 15 denoising steps inference

## Mask-Free Approaches

### OmniTry
- **Paper**: [2508.13632](https://arxiv.org/pdf/2508.13632)
- **Conference**: NeurIPS 2025
- **Strategy**: Extends VTON beyond clothes to any wearable (jewelry, accessories)
- **Key innovations**:
  - Two-staged training pipeline
  - Traceless erasing for avoiding shortcut learning
  - Inpainting-based re-purposing strategy
  - Masked full-attention for identity transferring
- **Benchmark**: 12 wearable object categories

### SMF-VTO (Style-Instructed Mask-Free)
- **Paper**: [2603.29587](https://arxiv.org/pdf/2603.29587)
- **Strategy**: Mask-free with text instructions for style control
- **Key innovations**:
  - Attention-guided loss for spatial ambiguity
  - Reference positional embedding module
  - Text-based style prompts for garment type/appearance control

## Multi-View / 3D Approaches

### VTON 360
- **Paper**: [2503.12165](https://arxiv.org/abs/2503.12165)
- **Strategy**: 3D VTON supporting any viewing direction
- **Key innovations**:
  - Pseudo-3D pose representation using SMPL-X normal maps
  - Multi-view spatial attention mechanism
  - Multi-view CLIP embedding with camera information

## Commercial / Industrial Systems

### Tstars-Tryon 1.0
- **Paper**: [2604.19748](https://arxiv.org/abs/2604.19748)
- **Deployment**: Taobao App, millions of users
- **Capabilities**:
  - Robust to extreme poses, illumination, motion blur
  - Multi-image composition (up to 6 reference images)
  - 8 fashion categories
  - Near real-time inference

### StageAttn-VTON
- **Paper**: [2026](https://www.mdpi.com/2076-3417/16/7/3609)
- **Strategy**: Stage-wise flow deformation with attention
- **Stages**: 3-stage decomposition for structural coherence
- **Key components**: Self-attention module for global dependency modeling

## Fit-Aware VTO

### FIT Dataset
- **Paper**: [2604.08526](https://arxiv.org/abs/2604.08526)
- **Contribution**: Large-scale dataset (1.13M triplets) with body/garment measurements
- **Strategy**:
  - GarmentCode for 3D garment generation
  - Physics simulation for realistic draping
  - Re-texturing framework for photorealistic synthesis
- **Task**: Fit-aware VTO (ill-fit cases)

## Summary of Architectures

| Architecture Type | Papers |
|-------------------|--------|
| DiT / Transformer | DiT-VTON, TED-VITON, PROMO, TEMU-VTOFF |
| Training-Free / PEFT | OmniVTON, OmniVTON++, DEFT-VTON |
| Mask-Free | OmniTry, SMF-VTO |
| 3D / Multi-View | VTON 360 |
| Commercial | Tstars-Tryon 1.0, StageAttn-VTON |
| Fit-Aware | FIT |

## Summary of Strategies

1. **DiT backbone adoption**: Major shift from U-Net to Diffusion Transformer
2. **Training-free generalization**: DDIM inversion + garment prior alignment
3. **Mask-free**: Eliminating segmentation mask dependency
4. **Universal VTO**: Beyond clothes to accessories, multi-category
5. **3D consistency**: Multi-view attention, SMPL-X normal maps
6. **Efficiency**: Flow matching, h-transform, self-reference
7. **Fit-awareness**: Body/garment measurement integration

## Related

[[Skills Index]]
[[RAG Index]]
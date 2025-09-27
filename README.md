# **Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback**  

[![Paper](https://img.shields.io/badge/arXiv-2506.03106-b31b1b.svg)](https://arxiv.org/abs/2506.03106)
[![Model](https://img.shields.io/badge/🤗%20Model-Critique_GRPO_Qwen3--8B-blue)](https://huggingface.co/xyingzhang/critique_grpo_math_4k_qwen3_8b_rollout7_self_critique_1_global_step_300)

![Method Overview](Introduction.png)

## Overview

Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language model (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. 

---

## Key Contributions

1. **Dual-Feedback Optimization**:
   - First framework to effectively combine natural language critiques with numerical rewards
   - Addresses the "plateau and forget" problem in RL fine-tuning

2. **Consistent Performance Gains**:
   - Outperforms baselines across 8 challenging benchmarks:
     - Mathematical reasoning (AIME, MATH)
     - STEM problem-solving
     - General reasoning tasks

- **Better Policy Exploration**: Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals:
  - **Higher entropy** does not always guarantee efficient learning from exploration.
  - **Longer responses** do not necessarily lead to more effective exploration.

- **Critique-Guided Refinements**: RL-finetuned models using Critique-GRPO demonstrate the ability to generate correct refinements for persistently failed problems, leveraging natural language critiques effectively.  

#### Critique-GRPO Framework
![Critique-GRPO Framework](Critique_GRPO.png)

![Three Types of Critique](Three_types_of_critique.png)


## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025critique,
  title={Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback},
  author={Zhang, Xiaoying and Sun, Hao and Zhang, Yipeng and Feng, Kaituo and Yang, Chao and Meng, Helen},
  journal={arXiv preprint arXiv:2506.03106},
  year={2025}
}

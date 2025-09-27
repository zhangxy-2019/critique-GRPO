# **Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback**  

[![Paper](https://img.shields.io/badge/arXiv-2506.03106-b31b1b.svg)](https://arxiv.org/abs/2506.03106)
[![Model](https://img.shields.io/badge/🤗%20Model-Critique_GRPO_Qwen3--8B-blue)](https://huggingface.co/xyingzhang/critique_grpo_math_4k_qwen3_8b_rollout7_self_critique_1_global_step_300)

![Method Overview](Introduction.png)

## Overview

Recent advances in reinforcement learning (RL) with numerical feedback have significantly enhanced LLM reasoning capabilities. However, we identify three key limitations:

1. **Performance plateaus** in later training stages
2. **Ineffective self-reflection** mechanisms
3. **Persistent failures** on challenging problems

**Critique-GRPO** is a novel online RL framework that combines:
- Natural language critiques
- Numerical rewards
- Advanced exploration techniques

**Core innovation**: Critique-GRPO operates through three phases (R1-ZERO Training Paradigm). Simultaneous learning from both initial responses and critique-guided refinements. 

## Released Resources

- **Model**: [Qwen3-8B Critique-GRPO](https://huggingface.co/xyingzhang/critique_grpo_math_4k_qwen3_8b_rollout7_self_critique_1_global_step_300)
  - Fine-tuned with self-critiquing capability
  - Optimized for mathematical reasoning
- **Code**: Initial version released, official version pending funding approval (The verl folder used by Critique-GRPO was lost during Git conflict resolution. We are actively reorganizing the codebase and will release an updated version shortly.)


#### Critique-GRPO Framework
![Critique-GRPO Framework](Critique_GRPO.png)

![Three Types of Critique](Three_types_of_critique.png)

### Quick Start
```bash
conda env create -f training_env.yml
conda activate critique-grpo
bash verl/examples/grpo_trainer/run_open_r1_math4k-qwen3-8b-critique_text_online.sh
```


## Acknowledgements
We gratefully acknowledge the open-source community and specifically thank [VERL](https://github.com/volcengine/verl), [LUFFY](https://github.com/ElliottYan/LUFFY).

## Citation
If you find our code useful, please cite:
```bibtex
@article{zhang2025critique,
  title={Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback},
  author={Zhang, Xiaoying and Sun, Hao and Zhang, Yipeng and Feng, Kaituo and Lu, Chaochao and Yang, Chao and Meng, Helen},
  journal={arXiv preprint arXiv:2506.03106},
  year={2025}
}"

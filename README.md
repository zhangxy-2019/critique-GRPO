# **Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback**  
[![Paper](https://img.shields.io/badge/arXiv-2506.03106-b31b1b.svg)](https://www.arxiv.org/abs/2506.03106)


---
The code is coming soon! Expect a release in 1–2 weeks. Stay tuned!

---

![Overview](Introduction.png)
=======

![Overview](figure1.png)

## Overview

Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language model (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. 

---

## Key Contributions

- **Improved Reasoning Performance**: Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across *eight challenging tasks*, including:
  - Mathematical reasoning
  - STEM problem-solving
  - General reasoning tasks

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

# Multi-agent Control Systems with Language Models
![output](https://github.com/user-attachments/assets/4af3f4c9-b3a6-48a1-b6a4-71a87867bd31#center)

**Description:** This repository contains the code for MAPPO (https://arxiv.org/abs/2103.01955), a multi-agent variant of PPO, on VMAS tasks (https://arxiv.org/abs/2207.03530) with a population of agents grounded by common language (https://arxiv.org/pdf/2304.03442). Hence, decision-making agents learned to utilize the capabilities of LLMs to complete various challenging tasks requiring an array of intelligent social behaviors.

In this implementation, we incorporate HuggingFace (https://huggingface.co/) and SBERT (https://sbert.net/index.html) to grant agents a simple RAG-based (https://arxiv.org/abs/2005.11401) memory and the ability to communicate with one another using natural language.


### Quickstart

Create and activate conda environment
```bash
$ conda create -n maclm python=3.11
$ conda activate maclm
```

Install requirements
```bash
$ pip install -r requirements.txt
```

Run training
```bash
$ python train.py --task=vmas/balance
```

View training results!

https://github.com/user-attachments/assets/1d791936-810e-48af-9ca1-59e5f6b25fc3


The code was implemented by Dom Huh, adopting and modifying from the code for PPO from CleanRL (https://github.com/vwxyzjn/cleanrl) and configs from BenchRL (https://github.com/facebookresearch/BenchMARL).

# Multi-agent Control Systems with Language Models
![output](https://github.com/user-attachments/assets/4af3f4c9-b3a6-48a1-b6a4-71a87867bd31#center)

**Description:** This repository contains the code that implements [MAPPO](https://arxiv.org/abs/2103.01955), a multi-agent variant of PPO, to train a population of decision-making agents that are able to work with natural language. Specifically, we use MARL tasks from [VMAS](https://arxiv.org/abs/2207.03530) and extend concepts of multi-agent language modeling from this [work](https://arxiv.org/pdf/2304.03442). In short, we train decision-making agents to learn to utilize the capabilities of LLMs to complete various challenging tasks requiring an array of intelligent social behaviors.

In this implementation, we use [HuggingFace](https://huggingface.co/) and [SBERT](https://sbert.net/index.html) to provide agents their own [RAG](https://arxiv.org/abs/2005.11401)-based memory and the ability to communicate with one another using natural language.


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

Run training (Modify conf/*/*.yaml if needed!)
```bash
$ python train.py task=vmas/balance
```

*Please adjust the configuration files in conf/\* accordingly depending on the system you are working with!

View training results!

https://github.com/user-attachments/assets/32b77065-7439-4dd4-a3b2-83a95ba350c3

https://github.com/user-attachments/assets/2d06b8e2-1496-4ab2-ab26-96292016b9b8

https://github.com/user-attachments/assets/1d791936-810e-48af-9ca1-59e5f6b25fc3


The code was implemented by Dom Huh, adopting and modifying from the code for PPO from CleanRL (https://github.com/vwxyzjn/cleanrl) and configs from BenchRL (https://github.com/facebookresearch/BenchMARL).

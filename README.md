<p align="center" width="100%">
</p>

<div id="top" align="center">

Weighted-Reward Preference Optimization for Implicit Model Fusion
-----------------------------
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/SLIT-AI/WRPO?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/SLIT-AI/WRPO?color=red" alt="Issues">

<h4> |<a href="https://arxiv.org/abs/2412.03187"> 📑 WRPO Paper</a> |
<a href="https://huggingface.co/AALF"> 🤗 HuggingFace Repo </a> |
<a href="https://github.com/yangzy39/WRPO"> 🐱 GitHub Repo </a> |
</h4>


</div>

## Overview
In this work, we introduce Weighted-Reward Preference Optimization (WRPO) for the implicit model fusion of heterogeneous open-source LLMs with diverse architectures and sizes, aiming to create a more capable and robust target LLM. 
As shown in Figure below, this objective introduces a fusion coefficient $\alpha$ that dynamically balances the internal reward of the preferred response y<sub>w<sub>s</sub></sub> from source LLMs and that of y<sub>w<sub>t</sub></sub> from the target during training. This approach enables the target LLM to transition smoothly from its distribution to align with that of the source LLMs.
<p align="center">
    <img src="./assets/wrpo_method.png" width="90%"> <br>
</p>

## Overall Results
In our experiments, we use LLaMA3-8B-Instruct as the target LLM. As for the source LLMs, we include ten advanced open-source models of varying architectures and sizes. We assess the performance of our models on three representative instruction-following benchmarks: MT-Bench, AlpacaEval-2, and Arena-Hard. Extensive experiments demonstrate that WRPO consistently outperforms existing knowledge fusion methods and various fine-tuning baselines.
<p align="center">
    <img src="./assets/main_results.png" width="70%"> <br>
</p>



## Requirements

This repository includes a [requirements file](requirements.txt) specifying the Python package versions used in our experiments. We utilized `Python 3.10` and `CUDA 12.2` for this work.

## Training Data Construction
We use one of the subset of UltraFeedback provided in [princeton-nlp/llama3-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) to construct our training dataset. We provide source LLM responses in [AALF/ultrafeedback_wrpo](https://huggingface.co//datasets/AALF/ultrafeedback_wrpo).

Our training dataset contains quadruples of (x, y<sub>w<sub>s</sub></sub>, y<sub>w<sub>t</sub></sub>, y<sub>l</sub>), where y<sub>w<sub>s</sub></sub> is a response from Source the LLMs, y<sub>w<sub>t</sub></sub> and y<sub>l</sub> are responses from the Target LLM.


### Target & Source LLMs
The Target and Source LLMs, along with their corresponding Huggingface IDs, are listed below:

| **Models**                   | **Huggingface ID**                                                                                                |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Target (LLaMA-3-8B-Instruct) | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)                 |
| Mistral-Large-Instruct-2407  | [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407)             |
| Gemma2-27B-IT                | [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)                                             |
| Qwen2-72B-Instruct           | [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)                                         |
| LLaMA3-70B-Instruct          | [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)               |
| Gemma2-9B-IT                 | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)                                               |
| Internlm2.5-20B-Chat         | [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)                             |
| DeepSeek-V2-Chat             | [deepseek-ai/DeepSeek-V2-Chat-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628)                     |
| DeepSeek-Coder-V2-Instruct   | [deepseek-ai/DeepSeek-Coder-V2-Instruct-0724](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724) |
| Yi-1.5-34B-Chat              | [01-ai/Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat)                                             |
| Phi-3-medium-4k-instruct     | [microsoft/Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)                   |

### Construction of y<sub>w<sub>s</sub></sub>
1. For each prompt in the [Ultrafeedback](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) dataset, we sample five responses from each source LLM. This was done using top-p sampling (p=0.95) with a temperature of 0.8, following the pipeline in the [SimPO repository](https://github.com/princeton-nlp/SimPO). 

2. We employ [ArmoRM-LLaMA3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) as the reward model to score and rank these responses. The highest-scoring response across all source models is selected as one of the preferred responses, y<sub>w<sub>s</sub></sub>. 


### Construction of y<sub>w<sub>t</sub></sub> and y<sub>l</sub>

1. The dataset is split into two parts: one-third of the data instances are randomly sampled for supervised fine-tuning (SFT), while the remaining instances are used for preference optimization, as detailed in our paper.

2. We apply Supervised Fine-Tuning (SFT) on the set of y<sub>w<sub>s</sub></sub> using first part of the dataset.

3. We then generate five samples from the SFT model using the remaining dataset. The response with the highest score is labeled as y<sub>w<sub>t</sub></sub>, while the lowest-scoring response is regarded as y<sub>l</sub>. 


Below is an example instance of our dataset, where "chosen" is a list containing [y<sub>w<sub>s</sub></sub>, y<sub>w<sub>t</sub></sub>], and "rejected" is a list containing [y<sub>l</sub>].

```
{
    "prompt_id": "3ebac2832721f4ef8e9ead1bb19e251e5d21d60dbd9f89cae931fe4aac900058",
    "prompt": {
        "content": "Given the task definition and input, reply with output. The provided file includes inquiries about restaurants in Finnish, and we ask you to translate those to English language. Please bear in mind the following guidelines while doing the translation: 1) We are looking for the most naturally written and formal form of each sentence in your language. We are *NOT* looking for colloquial forms of the sentence. We are looking for formal form which is how you would type your queries in a text-based virtual assistant. 2) The words between quotation marks *SHOULD NOT* be translated. We expect you to keep those values intact and include the quotation marks around them as well. 3) The fully capitalized words like DATE_0, or DURATION_0 *SHOULD NOT* be translated. Please keep them as they are in the translations. 4) Please do not localize measurement units like miles to kilometers during your translation. miles should be translated to its equivalent in your language. 6) Note the input is all lowercased except for fully capitalized special placeholders (e.g. NUMBER, DATE, TIME). Please do the same in your translations.\n\nkerro minulle \" dave 's seafood shack \" -ravintolan sijainti.\n",
        "role": "user"
    },
    "chosen": [
        {
            "content": "Tell me the location of the \"dave's seafood shack\" restaurant.",
            "role": "assistant"
        },
        {
            "content": "Here is the translated output:\n\nTell me the location of \"Dave's Seafood Shack\".",
            "role": "assistant"
        }
    ],
    "rejected": [
        {
            "content": "Here's the translation:\n\nTell me the location of \"Dave's Seafood Shack\".",
            "role": "assistant"
        }
    ]
}
```
## Training configurations

Our training code is built upon the [alignment-handbook repo](https://github.com/huggingface/alignment-handbook) and [SimPO repo](https://github.com/princeton-nlp/SimPO). We provide configuration files for both training stages, designed for an environment with 8x80G A800 GPUs. You may need to adjust `num_processes` and `per_device_train_batch_size` based on your specific computational setup.

### Commands

* To run Target-SFT:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file training_configs/deepspeed_zero3.yaml scripts/run_sft.py training_configs/llama-3-8b-instruct-sft.yaml
```

* To run Target-SFT-WRPO:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file training_configs/deepspeed_zero3.yaml scripts/run_wrpo.py training_configs/llama-3-8b-instruct-sft-wrpo.yaml
```

## Citation
```
@inproceedings{
    yang2025weightedreward,
    title={Weighted-Reward Preference Optimization for Implicit Model Fusion},
    author={Ziyi Yang and Fanqi Wan and Longguang Zhong and Tianyuan Shi and Xiaojun Quan},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```

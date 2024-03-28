---
layout: post
title:  "Recent Read Papers and Summaries"
---
This blog records the paper I recently read, I will try to partition them by topics.

[DPO](https://arxiv.org/abs/2305.18290)
**Summary**: This paper is very interesting! It has the following contribution:
It offers a framework where people can directly align the model according to the human preference data instead of the approach in RLHF, which requires training a reward preference function, and using this function to align to the preference data.
It offers sound theoretical proof and reasoning that using this method, we can automatically get a reward function for free during the training process, and this reward function is unique up to some equivalent conditions.
The theoretical framework also pinpoints why the popular method PPO is unstable in training.

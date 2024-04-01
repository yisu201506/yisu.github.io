---
layout: post
title:  "Recent Read Papers and Summaries"
---
This blog records the paper I recently read, I will try to partition them by topics.

## [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
**Summary**: This paper is very interesting! It has the following contribution:
It offers a framework where people can directly align the model according to the human preference data instead of the approach in RLHF, which requires training a reward preference function, and using this function to align to the preference data.
It offers sound theoretical proof and reasoning that using this method, we can automatically get a reward function for free during the training process, and this reward function is unique up to some equivalent conditions.
The theoretical framework also pinpoints why the popular method PPO is unstable in training.

## [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
**Summary**: This paper introduced a policy gradient objective, which makes the RL training more stable. Instead of $E(\pi_{\theta}(a_t\|s_t)A_t)$, it optimizes $E(\min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t))$, where $r_t(\theta) = \pi(a_t\|s_t)/\pi_{old}(a_t\|s_t)$, this gives a constraint such that the policy function won’t increase forever, and does not perform reward hacking. Exploration can be achieved by adding an entropy bonus on this function. This function proves to learn faster and converge faster compared with other methods. This method is also used later in the traditional LLM training.

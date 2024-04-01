---
layout: post
title:  "Recent Read Papers and Summaries"
---
This blog records the paper I recently read, I will try to partition them by topics.

## [InstructGPT](https://arxiv.org/abs/2203.02155)
**Summary**: Historically, the language model trained on large corpses of texts using next word prediction is not so good at following instructions, answering questions or other preferred way that humans want it to perform. This paper provides a method or rather a sequence of steps to align the language model to the human preference in texts and values in it.

Specifically, there are three steps:
1. First finetune a previously pretrained model on a manually collected prompt-response data set. These prompts are initially collected through researcher’s instruction, and once the initial model is in beta API, it is then partially collected through the API distribution.
2. Once we have a supervised fine-tuning model, use it to sample K completions of the same prompt (the prompt are either from beta API or labeler), and ask labelers to rank these K prompts. These preferences form K choose 2 pairwise data points. Then pass these data points to SFT model with last layer removed to form a set of embeddings. Use this embeddings to train a reward model by the loss function $-\dfrac{2}{K(K-1)}E_{(x, y_w, y_l)\in D}[\log(\sigma(r_{\theta}(x, y_w) - r_{\theta}(x,y_l)))]$, this is the binary cross entropy on the difference of two completions.
3. Finally, fix this trained reward model, and make the language model trainable. Sample the prompts from the API, pass through the language model, and use the RM to give a score/reward to each response. We use PPO to train this human RL feedback. And the final result will be the Instruct GPT model. Note that, we throw in a negative KL divergence term between the SFT language policy and the learned language policy, so that the learned policy won’t be too far away from original policy due to reward hacking

The test shows that this model is preferred compared with GPT3, and generalizes well to the unseen dataset, such as new language, in the RLHF process, and also generalizes well to the labelers who did not participate in the training data labeling process. With proper preference definition, this can train an LLM which caters towards one specific set of preferences.

## [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
**Summary**: This paper followed the approach from [OpenAI’s RLHF](https://arxiv.org/abs/2009.01325), and used data that are created towards being helpful and harmless, which are interpreted by the human reviewers themselves. Then it finds that the performance has trade off between helpfulness and harmlessness, but does not have trade off on other capabilities. Majority of the paper talks about how to collect the data and the evaluation process. One thing worth mentioning is the calibration study on the preference models (reward models). It shows that the preference models has pretty good calibration. And thus we can trust the rewards it assigned in the RL process.

## [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
**Summary**: This paper is very interesting! It has the following contribution:
It offers a framework where people can directly align the model according to the human preference data instead of the approach in RLHF, which requires training a reward preference function, and using this function to align to the preference data.
It offers sound theoretical proof and reasoning that using this method, we can automatically get a reward function for free during the training process, and this reward function is unique up to some equivalent conditions.
The theoretical framework also pinpoints why the popular method PPO is unstable in training.

## [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
**Summary**: This paper introduced a policy gradient objective, which makes the RL training more stable. Instead of $E(\pi_{\theta}(a_t\|s_t)A_t)$, it optimizes $E(\min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t))$, where $r_t(\theta) = \pi(a_t\|s_t)/\pi_{old}(a_t\|s_t)$, this gives a constraint such that the policy function won’t increase forever, and does not perform reward hacking. Exploration can be achieved by adding an entropy bonus on this function. This function proves to learn faster and converge faster compared with other methods. This method is also used later in the traditional LLM training.

## [Gradient Low-rank Projection (GaLore)](https://arxiv.org/abs/2403.03507)
Summary: if one looks at the gradient decent process as a curve in the weight space, one can think of Galore method is to use a piecewise linear function to approximate this curve. These linear segments live in a subspace of the original weight space, and thus can be parametrized by a smaller weight matrix. The contribution of this paper is that it chooses these piecewise linear segments to be the subspace formed by the first $r$ principal components of the weight matrices, and these projection matrices (to these subspaces) are modified every $N$ steps. Here $r$ and $N$ are hyperparameters to be tuned. This method is particularly interesting is that this may follow the original gradient descent curve instead of some contrived parametrization such as LoRA.

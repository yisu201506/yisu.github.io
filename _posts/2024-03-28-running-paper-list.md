---
layout: post
title:  "Recent Read Papers and Summaries"
---
This blog records the paper I recently read, I will try to partition them by topics. I use the number of 👍 to represent how much I enjoy the paper, and it **DOES NOT** reflect the actual theoretical and practical contribution of the paper. All the graphs are referred through the title link unless otherwise specified.

# LLM Papers

## [Mixture of Experts (MoE)](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe) 👍

**Summary**: Mixture of Experts is an ensemble method, in which not all experts are activated to compute the activation. Specifically, the application in LLM is that one feed forward network (FFN) in each transformer block is replaced with several expert FFNs, and a gate/router is used to decide the best experts to go to in training and inference. 
![Screen Shot 2024-04-01 at 8 34 56 PM](https://github.com/yisu201506/yisu201506.github.io/assets/12384424/a3aa903f-9de9-498b-9028-5d310d0b3480)

There are a few advantages:
1. This will make pretraining and inference a lot faster for the same amount of parameters. Due to the ensemble nature, the result seems quite good.
2. Different experts seem to specialize on different things, and thus more explanable compared with typical transformer blocks.

However, there are quite a few drawbacks. 
1. Although not using all experts are used in the inference, one still needs to load these all experts in the memory, and thus requires a high memory machine.
2. Finetuning processes tends to overfit, and thus requires higher dropout or adding noises in the routers.
3. The infra serving MoE is a bit more complexed as typical batching may not work well since the activation of experts are sparse (not all experts are activated).

To be honest, since MoE takes more memory to run what is equivalently a smaller model, I do not think this trade-off would be a trend in the research and industry, where memory will be limited in the personal device.

## [InstructGPT](https://arxiv.org/abs/2203.02155) 👍👍
**Summary**: Historically, the language model trained on large corpses of texts using next word prediction is not so good at following instructions, answering questions or other preferred way that humans want it to perform. This paper provides a method or rather a sequence of steps to align the language model to the human preference in texts and values in it.

Specifically, there are three steps:
1. First finetune a previously pretrained model on a manually collected prompt-response data set. These prompts are initially collected through researcher’s instruction, and once the initial model is in beta API, it is then partially collected through the API distribution.
2. Once we have a supervised fine-tuning model, use it to sample K completions of the same prompt (the prompt are either from beta API or labeler), and ask labelers to rank these K prompts. These preferences form K choose 2 pairwise data points. Then pass these data points to SFT model with last layer removed to form a set of embeddings. Use this embeddings to train a reward model by the loss function $-\dfrac{2}{K(K-1)}E_{(x, y_w, y_l)\in D}[\log(\sigma(r_{\theta}(x, y_w) - r_{\theta}(x,y_l)))]$, this is the binary cross entropy on the difference of two completions.
3. Finally, fix this trained reward model, and make the language model trainable. Sample the prompts from the API, pass through the language model, and use the RM to give a score/reward to each response. We use PPO to train this human RL feedback. And the final result will be the Instruct GPT model. Note that, we throw in a negative KL divergence term between the SFT language policy and the learned language policy, so that the learned policy won’t be too far away from original policy due to reward hacking

The test shows that this model is preferred compared with GPT3, and generalizes well to the unseen dataset, such as new language, in the RLHF process, and also generalizes well to the labelers who did not participate in the training data labeling process. With proper preference definition, this can train an LLM which caters towards one specific set of preferences.

## [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) 👍
**Summary**: This paper followed the approach from [OpenAI’s RLHF](https://arxiv.org/abs/2009.01325), and used data that are created towards being helpful and harmless, which are interpreted by the human reviewers themselves. Then it finds that the performance has trade off between helpfulness and harmlessness, but does not have trade off on other capabilities. Majority of the paper talks about how to collect the data and the evaluation process. One thing worth mentioning is the calibration study on the preference models (reward models). It shows that the preference models has pretty good calibration. And thus we can trust the rewards it assigned in the RL process.

## [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) 👍👍👍
**Summary**: This paper is very interesting! It has the following contribution:
It offers a framework where people can directly align the model according to the human preference data instead of the approach in RLHF, which requires training a reward preference function, and using this function to align to the preference data.
It offers sound theoretical proof and reasoning that using this method, we can automatically get a reward function for free during the training process, and this reward function is unique up to some equivalent conditions.

The derivation is also mathmatically pleasing. Suppose we have trained a reward function $r(x, y)$, then the object function of the reinforcement learning step by sampling the reward function is
$\max_{\pi_{\theta}}(E_{x \sim D, y \sim \pi_{theta}(y\|x)}[r_{\phi}(x,y)] -\beta D_{\text{KL}}[\pi_{\theta}(y\|x)||\pi_{\text{ref}}(y\|x)])$. It can be shown that the optimal policy $\pi_r(y\|x)$ can be written analytically as $\pi_{r}(y|x) = \dfrac{1}{Z(x)}\pi_{\text{ref}}(y|x) \exp(\dfrac{1}{\beta}r(x, y))$, where $Z(x)$ is a function of $x$ only. And one can rearrange the above equation for $r(x,y)$, and get $r(x, y) = \beta\log \dfrac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)$.

We can plug in the above equation back to the objective function that we use to approximate the reward function $r_{\phi}(x,y)$, namely $-E_{(x, y_w, y_l) \sim D}[\log\sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l))]$, and obtain the DPO objective function
$L_{DPO}(\pi_{\theta}\pi_{\text{ref}}) = - E_{(x, y_w, y_l) \sim D}\Big[\log\sigma\Big(\beta\log\dfrac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \dfrac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big]$

After we train this model, we can get the reward function for free as $r(x, y) = \beta\log \dfrac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)}$. It can be show that $r(x, y)$ is unique up to an addition of a function of $x$. The theoretical framework also pinpoints why the popular method PPO is unstable in training.

## [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) 👍
**Summary**: This paper introduced a policy gradient objective, which makes the RL training more stable. Instead of $E(\pi_{\theta}(a_t\|s_t)A_t)$, it optimizes $E(\min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t))$, where $r_t(\theta) = \pi(a_t\|s_t)/\pi_{old}(a_t\|s_t)$, this gives a constraint such that the policy function won’t increase forever, and does not perform reward hacking. Exploration can be achieved by adding an entropy bonus on this function. This function proves to learn faster and converge faster compared with other methods. This method is also used later in the traditional LLM training.

## [Gradient Low-rank Projection (GaLore)](https://arxiv.org/abs/2403.03507) 👍👍👍
**Summary**: if one looks at the gradient decent process as a curve in the weight space, one can think of Galore method is to use a piecewise linear function to approximate this curve. These linear segments live in a subspace of the original weight space, and thus can be parametrized by a smaller weight matrix. The contribution of this paper is that it chooses these piecewise linear segments to be the subspace formed by the first $r$ principal components of the weight matrices, and these projection matrices (to these subspaces) are modified every $N$ steps. Here $r$ and $N$ are hyperparameters to be tuned. This method is particularly interesting is that this may follow the original gradient descent curve instead of some contrived parametrization such as LoRA.

## [BitNet](https://arxiv.org/abs/2310.11453) 👍
**Summary**: Contrast to post training quantization. This work trained a network with a one bit weight matrix (sign of the original weight), and quantized activation. During the training process, one would still use a latent weight matrix (the original weight matrix), but this weight matrix is further quantized by the operations of taking sign, min, max. During backward propagation, these operations are almost differentiable, and thus can do backward propagation.The performance is worse than typical transformer, but the energy consumption is a fraction of that for full transformers. (To be honest, I don’t see that why energy consumption should be prioritized at this stage of research)

## [OneBit](https://arxiv.org/abs/2402.11295) 👍
**Summary**: Use a teacher-student, to distill a full precision model to a one-bit weight plus two full precision vectors which corresponds to the sign matrix of the weight matrix W and the first principal component of the matrix W. The training data is from the teacher model’s output, and the objective function is both cross entropy between the teacher model’s probabilistic output and the student's probabilistic output and the L2 distance between the normalized hidden layers.

## [LoRA](https://arxiv.org/abs/2106.09685) 👍
**Summary**:  Training a LM requires keeping model weights in memory, and keep updating them using stochastic gradient descent or a variation such as Adam. These gradients and related optimization states are typically parametrized in the same space as the original weight space, and thus requires the same memory to store. Geometrically speaking, one can think each weight point as point in the total weight space, and SGD is just drawing a curve that contains these points from the initial weights to the point which gives lower loss.

The contribution of this paper is to give a (parametrized) subspace of the original weight space, and the SGD curve can only live on that space. This subspace maintains substantially fewer parameters (up to 1/10,000 of the original weights), and thus saves a lot of memory. The downside of this approach is that the SGD on the parameterized subspace may not lead to the global minimum as it is only a slice of the original space. Therefore , it suffers from performance losses if we use this method to do an entire training. However, if we already have a pretrained model, which is on a local minimum, using this method to fine-tune a model may not lead to much performance loss. Ideally, we want to choose a subspace or a set of subspaces which parameterize the original SGD curve more closely than LoRA.

## [vLLM](https://arxiv.org/abs/2309.06180), [Blog](https://blog.vllm.ai/2023/06/20/vllm.html) 👍👍👍
**Summary**: This paper introduces an efficient LLM inference and serving methods. Specifically, model inference typically has a maximum token, and during inference the contiguous space for these maximum token length is reserved in GPU DRAM, and potentially unused. These caused memory waste up to 80% of the available memory (excluding the memory used for storing the model parameters). PagedAttention, as in the picture below breaks up these sequences into smaller blockers and store each of these logically contiguous blocks in the (maybe noncontiguous) physical blocks using a block table. This way, almost all blocks are made sure not to waste space. Because these blocks are small, different decoding strategies such as parallel sampling and beam search can fully leverage sharing of these blocks, and thus further reduce the GPU memory usage.
![Screen Shot 2024-03-31 at 12 39 49 PM](https://github.com/yisu201506/yisu201506.github.io/assets/12384424/5f5781a5-f3bd-4a2c-bf66-9f095d1dd60a)

## [LMSYS Arena ](https://arena.lmsys.org/) 👍👍👍
This is an interesting website for different LLM to dual with each other.

# Image Related Papers

## Diffusion Models 👍👍👍
* [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html#mjx-eqn%3Aeq%3A78)
* [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)

**Summary**: The problem that we are trying to solve here is that given an empirical dataset, how can we create a model about that dataset, and use it to generate samples from similar to the dataset. This process can be conditioned on some external data as well. Specifically, the authors of these blogs are more concerned with high fidelity image generation.

The first blog focuses on the diffusion model from maximum likelihood methods. Specifically, given dataset $D$, we want to find the underlying distribution such that it solves 
\begin{equation}
$argmax\prod_{x\in D} p(x) = argmax\sum_{x\in D} \log p(x) \approx E(\log p(x)) \geq \text{Evidence Lower Bound\}$
\end{equation}

Therefore we can maximize the evidience lower bound to model the original dataset. This modeling process involves first keep adding noise to the original data for $T$ steps until the resulting data is roughly standard Guassian distribution. Note that this step is more or less deterministic. And then we create the dataset by using the timestep $t$, and the latent image at $t$ to predict the noise added to original data to obtain the latent data at $t$.

The second blog focuses on how we use score, which is the gradient of the log probability $\nabla\log p(x)$, and Langevin dynamics, a discrete version of stochastic differential equation to dynamically morph an arbitrary prior distribution to the dataset distribution based on the proximated score.

These two approaches are connected via Tweedie’s Formula. Furthermore, we can inject conditions to these models so that we can generate a guided model based on some image and text cues.


# Other ML topics
## [Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440)
**Summary**: This paper shows that the learned cosine similarity in the learned embedding may not reflect the semantic proximity of the samples. The author uses the matrix factorization model that is often used in Netflix and Amazon to illustrate the points, In fact, for one popular object function that is interpreted to be denoising, applying a diagonal/normalization matrix would not change the optimality of the solution, but the cosine similarity is completely destroyed for item - item relationship. In fact, it is an identity matrix. 

However, if the solution space is a dense manifold, then it is difficult to make the solution exactly on the boundary (Probability = 0 if the boundary has zero measure). We should look more closely at the solution space.




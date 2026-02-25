## Chapter 9 - Reinforcement Learning

* Frozen Lake RL with an MLP

## Deep Q Learning, the loss, and the bellman equation

Why Q-Pred Depends on (s, a) but the Target Uses Only max Q(s₁)

I guess if you have negative or zero Q-values, those naturally get beaten by positive ones, and positive ones get beaten by even higher ones. Higher values only appear when the agent has done something good that led to rewards, so the value landscape naturally forms a ranking: bad actions stay low, neutral actions hover near zero, and useful actions rise above the rest.

Why q_pred = Q(s, a):

The predicted value comes from the specific state–action pair you actually took. The network made a concrete guess about that exact choice, so this is the only number you can legitimately correct. The loss must update the model’s belief about the action you just performed, because you have direct evidence for whether that choice turned out good or bad.

Why target = r + γ · max Q(s₁):

The target does not try to match the action you will take next. Instead, Q-learning uses the best possible future value from the next state, regardless of which action was chosen during exploration. This reflects the Bellman optimality principle: the value of today’s action should equal the immediate reward plus the best achievable future reward. Using max Q(s₁) pulls Q(s, a) upward only when the next state contains promising futures, and pushes it downward when the next state is unrewarding. Over time, this makes only the genuinely good paths accumulate high values.



## Reinforcement Learning

* Q learning (value learning)
* Policy: PPO, DPO, GRPO

## RLHF

* GPT ( RLHF, policy )

## Reverse Engineering

* https://medium.com/@aalokpatwa/dpo-and-sft-tuning-of-gpt-2-in-pure-pytorch-d025dff6333f
* https://github.com/aalokpatwa/dpo
* 

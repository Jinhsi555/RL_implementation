# 强化学习算法实现

本项目使用PyTorch实现了多种强化学习算法，包括DQN、REINFORCE、Actor-Critic、PPO和SAC。每个算法都有对应的Python文件，可以直接运行进行训练和测试。代码简单容易理解，配有详细的注释，帮助初学者快速上手。

另外还有一个 jupyter notebook 文件，包含对 Reinforcement Learning 基础的讲解。

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- OpenAI Gym 0.26+
- NumPy
- Matplotlib

## 安装

运行以下命令安装依赖：
pip install -r requirements.txt

## RL 基础 tutorial

* 多臂老虎机问题作为引子探讨探索与利用平衡
* 增量式期望更新，为 Monte-Carlo 方法和时序差分（TD）做铺垫
* $\epsilon\text{-贪心算法}$
* 上置信界 UCB 算法
* MDP 马尔可夫决策过程
* 价值函数
* 贝尔曼期望方程
  * 状态价值函数的贝尔曼期望方程
  * 动作价值函数的贝尔曼期望方程
* 贝尔曼最优方程
* Monte-Carlo 方法

## 实现算法

- **DQN**: 基于值函数的off-policy算法，使用经验回放和目标网络
- **REINFORCE**: 基于策略梯度的on-policy算法
- **Actor-Critic**: 结合策略梯度和值函数估计的算法
- **PPO**: 近端策略优化算法，通过限制策略更新幅度来提高稳定性
- **SAC**: 软演员-评论家算法，结合熵正则化以实现更好的探索

## Reference

* [动手学强化学习（上海交大 RL 教程）](https://github.com/boyu-ai/Hands-on-RL.git)
* [Easy-RL 李宏毅老师《深度强化学习》笔记](https://github.com/datawhalechina/easy-rl.git)
* [强化学习的数学原理 西湖大学赵世钰老师](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning.git) [视频地址](https://www.bilibili.com/video/BV1sd4y167NS/?vd_source=8d937f1d6aa4e3f9b84f393d40fc10e8)

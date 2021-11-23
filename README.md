# Project Work of ELEC-E8125
In the project work, we will implement and apply some more advanced RL algorithms in continuous control tasks. The project work includes two parts. First, two vastly used reinforcement learning algorithms, **TD3** (https://arxiv.org/pdf/1802.09477.pdf) and **PPO** (https://arxiv.org/abs/1707.06347), will be implemented. For this part, we will offer the base code so you can start easily. After finishing this part, you can train a policy to balance a cart pole and to control an halfcheetah running forward. 

In the second part, you need to read some research papers and implement their proposed algorithms based on the code finished in Part I. The candidate algorithms in Part II include:

- MBPO (https://arxiv.org/abs/1906.08253): How to use model-based approach to improve sample efficiency?
- REDQ (https://arxiv.org/abs/2101.05982): How to significantly improve sample efficiency of model-free RL with ensembles?
- SAC (https://arxiv.org/abs/1801.01290, https://arxiv.org/abs/1805.00909): These two papers offer you a view of how to treat RL as probabilistic inference.
- TD3_BC (https://arxiv.org/abs/2106.06860): An offline RL method that learns policy without interacting with the environment.

According to your preference, you can choose one of them to understand the paper and to implement the algorithm. For the listed algorithms, we will offer you the reference training curve. If you are interested in other algorithms, you can also choose them in Part II, but we can not offer much help in implementing those algorithms. 

This project work is supposed to be done in groups of **2 students**. If you need to find a partner for the project, please join the `project` channel on Slack and advertise yourself. The deadline for both the course project and the alternative project is **05.12.2021, 23:55.** 

#### Alternative Project

Alternatively, students can also propose their own project topic. This option is mainly aimed at PhD students that want to apply Reinforcement Learning to their own field, but Master's students are also encouraged. 

**Alternative project topics are individual.** The project proposal needs to be submitted and will be evaluated by the staff of the course, and can be started once the project is approved. 

The deadline for the alternative course project proposal is **29.10.2021 at 23:55.**

Grading
The course project grade accounts for **30%** of the final grade of the course.

## Install dependencies
1. If you need to build a new working environment (e.g., you need to change to a GPU machine in Aalto):
    ```python
    cd <your working path>
    virtualenv rl_proj  #create virtual env
	source  ./rl_proj/bin/activate #activate virtual env
	pip install --upgrade -r requirements.txt #install packages
	<Run your code>
	# if you want to deactivate the virtual env
	deactivate
	```
	
  2. If you already have your working environment, you can just run

    ```python
    pip install --upgrade -r requirements.txt
    ```

## Visualize your training
In our project, we use Weights & Bias to visualize our training. It's a tool similar to TensorBoard. You are free to use TensorBoard if comfortable, but the code base is using Weight & Bias. Here is the tutorial: https://docs.wandb.ai/quickstart. To use Weights & Bias, you need to create an account, then everything should be straightforward.



Also, to expert training plots, you need to first edit the `eval/returns` panel and change the x-axis from `Step` to `eval/timesteps`. Then click `more` to `export panel` with `PNG`. After setting proper plot name and size, you can `Download PNG`.

## How to run the code
After filling the TODOs, you can just run the code with:
```python
# under virtualenv
python3 part1/ppo.py
# or
python3 part1/td3.py
```

Please also set the correct FLAGs, e.g., if you want to save model, you can run like this:

```python
python3 part1/ppo.py --save_model
```


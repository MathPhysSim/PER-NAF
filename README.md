# PER-NAF

An implementation of the Normalized Advantage Function Reinforcement Learning Algorithm with Prioritized Experience Replay

## Summary

* The original paper of this code is: <https://arxiv.org/abs/1603.00748>
* The code is mainly based on: <https://github.com/carpedm20/NAF-tensorflow/>
* Additionally I added the prioritized experience replay: <https://arxiv.org/abs/1511.05952>
* Using the OpenAI baseline implementation: <https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py>

Thanks openAI and Kim!

## Some Advices from experience in RL

* Normalize the state and action space as well as the reward is a good practice
* Visualise as much as possible to get an intuition about the method as possible bugs
* If it does not make sense it is a bug with very high probability

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd PER-NAF
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the package in editable mode:

    ```bash
    pip install -e pernaf
    ```

## Benchmarks

The project has been fully migrated to **PyTorch**.
The project includes benchmark scripts to verify the NAF agent's performance on standard Gymnasium environments.

### Pendulum-v1

Trains the NAF agent on the continuous Pendulum task.

```bash
python train_pendulum_benchmark.py
```

Results (plots and stats) will be saved to the `results/` directory.

### Results

After running the benchmarks, check `results/` for training curves and evaluation plots.

#### training stats

![Training Statistics](results/pendulum_training_stats.png)

#### evaluation plot

![Evaluation Plot](results/pendulum_eval_plot.png)

### Algorithm Comparison

We compared three variants of the NAF agent on `Pendulum-v1` (40k steps).
The plot below shows the **Episode Reward** (Moving Average):

1. **NAF (Baseline)**: Standard NAF with Single Q-Learning.
2. **NAF2 (Double Q)**: NAF with Double Q-Learning.
3. **PER-NAF2**: Double Q-Learning + Prioritized Experience Replay.

![Benchmark Comparison](results/naf_vs_naf2_rewards.png)

## Usage

To run the main training loop:

```bash
python naf2.py
```

## Structure

* `naf2.py`: Main entry point for training NAF agents.
* `simulated_environment_final.py`: Simulated environment for the agent (AWAKE electron beam).
* `pernaf/`: core package containing Algorithm implementation (PER-NAF).

## Algorithm Details

This implementation features four variants of the NAF algorithm, designed to improve stability and sample efficiency.

### 1. NAF (Normalized Advantage Function)

NAF enables Q-learning in continuous action spaces by decomposing the Q-function into a State-Value term $V(s)$ and a quadratic Advantage term $A(s,a)$:

$$ Q(s, a | \theta) = V(s | \theta) + A(s, a | \theta) $$
$$ A(s, a | \theta) = -\frac{1}{2} (a - \mu(s | \theta))^T P(s | \theta) (a - \mu(s | \theta)) $$

* **Optimization**: $\arg\max_a Q(s,a) = \mu(s)$, allowing analytic action selection.
* **Pros**: Efficient off-policy learning for continuous tasks.

### 2. NAF2 (NAF + Double Q-Learning)

Incorporates Clipped Double Q-Learning to address value overestimation bias. Two networks are trained, and the target value uses the minimum estimate:

$$ V_{target}(s') = \min ( V(s' | \theta'_1), V(s' | \theta'_2) ) $$

* **Pros**: Significantly improved stability and resistance to divergence.
* **Reference**: Used in **Model-free and Bayesian Ensembling Model-based Deep Reinforcement Learning for Particle Accelerator Control Demonstrated on the FERMI FEL**, Simon Hirlaender, Niky Bruchon. <https://arxiv.org/abs/2012.09737>

### 3. PER-NAF (Prioritized Experience Replay)

Replaces uniform sampling with prioritized sampling based on the TD-error $\delta$:

$$ P(i) \propto | \delta_i |^\alpha $$

* **Pros**: Focuses learning on "surprising" or difficult transitions, improving data efficiency.

### 4. PER-NAF2 (Combined)

Combines the stability of **Double Q-Learning** with the potential of **Prioritized Replay**.

* **Target**: Double Q min-target.
* **Sampling**: Prioritized by Double Q TD-error.
* **Pros**: **State-of-the-art** performance for this suite; high stability and very high sample efficiency.

### Summary Comparison

| Algorithm | Model-Free | Overestimation Safe? | Data Efficiency | Best Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **NAF** | Yes | No | Normal | Simple Baselines |
| **NAF2** | Yes | **Yes** | Normal | Unstable Environments |
| **PER-NAF** | Yes | No | **High** | Expensive Interactions |
| **PER-NAF2** | Yes | **Yes** | **Very High** | **Complex Control** |

## Reference

This code was used in the following publication:

**Sample-efficient reinforcement learning for CERN accelerator control**
Verena Kain, Simon Hirlander, Brennan Goddard, Francesco Maria Velotti, Giovanni Zevi Della Porta, Niky Bruchon, and Gianluca Valentino.
*Phys. Rev. Accel. Beams* 23, 124801 (2020)
<https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.23.124801>

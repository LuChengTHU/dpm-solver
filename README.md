# DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps

The official code for the paper [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) (**Neurips 2022 Oral**) by Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li and Jun Zhu.

--------------------

DPM-Solver is a fast dedicated high-order solver for diffusion ODEs with the convergence order guarantee. DPM-Solver is suitable for both discrete-time and continuous-time diffusion models **without any further training**. Experimental results show that DPM-Solver can generate high-quality samples in **only 10 to 20** function evaluations on various datasets.

![DPM-Solver](assets/intro.png)

<br />

# News
- **2022-10-26**. We have updated the **DPM-Solver v2.0**, a more stable version for high-resolutional image synthesis tasks. We have the following upgrades:
    - We support the discrete-time DPMs by implementing a picewise linear interpolation of $\log\alpha_t$ for the `NoiseScheduleVP`.
    
        We strongly recommend to use the new implementation for discrete-time DPMs, especially for high-resolutional image synthesis. You can set `schedule='discrete'` to use the corresponding noise schedule. We also change the mapping between discrete-time inputs and continuous-time inputs in the `model_wrapper`, which has a consistent converged results with the other solvers.
    - We change the API for `model_wrapper`, which is more easy to use.
    - We support **new algorithms** for DPM-Solver, which greatly improve the high-resolutional image sample quality by guided sampling.
        - We support both the noise prediction model $\epsilon_\theta(x_t,t)$ and the data prediction model $x_\theta(x_t,t)$. For the data prediction model, we further support the *dynamic thresholding* introduced by [Imagen](https://arxiv.org/abs/2205.11487).
        - We support both *singlestep* solver (i.e. Runge-Kutta-like solver) and *multistep* solver (i.e. Adams-Bashforth-like solver) for DPM-Solver, including order 1, 2, 3.

<br />

# Use DPM-Solver in your own code
It is very easy to combine DPM-Solver with your own diffusion models. We support both Pytorch and JAX code. You can just copy the file `dpm_solver_pytorch.py` or `dpm_solver_jax.py` (The JAX code is cleaning and will be released soon) to your own code files and import it.

In each step, DPM-Solver needs to compute the corresponding $\alpha_t$, $\sigma_t$ and $\lambda_t$ of the noise schedule. We support the commonly-used variance preserving (VP) noise schedule for both discrete-time and continuous-time DPMs:

- For discrete-time DPMs, we support a picewise linear interpolation of $\log\alpha_t$  in the `NoiseScheduleVP` class. It can support all types of VP noise schedules.

- For continuous-time DPMs, we support both linear schedule (as used in [DDPM](https://arxiv.org/abs/2006.11239) and [ScoreSDE](https://arxiv.org/abs/2011.13456)) and cosine schedule (as used in [improved-DDPM](https://arxiv.org/abs/2102.09672)) in the `NoiseScheduleVP` class.

Moreover, DPM-Solver is designed for the continuous-time diffusion ODEs. For discrete-time diffusion models, we also implement a wrapper function to convert the discrete-time diffusion models to the continuous-time diffusion models in the `model_wrapper` function.

<br />

## Unconditional Sampling by DPM-Solver
We recommend to use **3rd-order singlestep** DPM-Solver for the **noise prediction model**. Here is an example for discrete-time DPMs:

```python
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

## You need to firstly define your model and the extra inputs of your model,
## And initialize an `x_T` from the standard normal distribution.
## `model` has the format: model(x_t, t_input, **model_kwargs).
## If your model has no extra inputs, just let model_kwargs = {}.

## If you use discrete-time DPMs, you need to further define the
## beta arrays for the noise schedule.

# model = ....
# model_kwargs = {...}
# x_T = ...
# betas = ....

## 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time noise prediction model `model`
## to the continuous-time noise prediction model.
model_fn = model_wrapper(
    model,
    noise_schedule,
    total_N=betas.shape[0],
    model_kwargs=model_kwargs,
)

## 3. Define dpm-solver and sample by singlestep DPM-Solver.
## (We recommend singlestep DPM-Solver for unconditional sampling)
## You can adjust the `steps` to balance the computation
## costs and the sample quality.
dpm_solver = DPM_Solver(model_fn, noise_schedule)

## You can use steps = 10, 12, 15, 20, 25, 50, 100.
## Empirically, we find that steps in [10, 20] can generate quite good samples.
## And steps = 20 can almost converge.
x_sample = dpm_solver.sample(
    x_T,
    steps=15,
    order=3,
    skip_type="time_uniform",
    method="singlestep",
)
```

<br />

## Guided Sampling by DPM-Solver
We recommend to use **2nd-order multistep** DPM-Solver for the **data prediction model** (by setting `predict_x0 = True`), especially for large guidance scales. Here is an example for discrete-time DPMs:

```python
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

## You need to firstly define your model and the extra inputs of your model,
## And initialize an `x_T` from the standard normal distribution.
## `model` has the format: model(x_t, t_input, **model_kwargs).
## If your model has no extra inputs, just let model_kwargs = {}.

## If you use discrete-time DPMs, you need to further define the
## beta arrays for the noise schedule.

## For classifier guidance, you need to further define a classifier function
## and a guidance scale.

# model = ....
# model_kwargs = {...}
# x_T = ...
# betas = ....
# classifier = ...
# classifier_kwargs = {...}
# guidance_scale = ...

## 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time noise prediction model `model`
## to the continuous-time noise prediction model.
model_fn = model_wrapper(
    model,
    noise_schedule,
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    total_N=betas.shape[0],
    model_kwargs=model_kwargs,
    classifier_kwargs=classifier_kwargs,
    condition_key="y",
)

## 3. Define dpm-solver and sample by singlestep DPM-Solver.
## (We recommend singlestep DPM-Solver for unconditional sampling)
## You can adjust the `steps` to balance the computation
## costs and the sample quality.

dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True)

## If the DPM is defined on pixel-space images, you can further
## set `thresholding=True`. e.g.:

# dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True,
#   thresholding=True)


## You can use steps = 10, 12, 15, 20, 25, 50, 100.
## Empirically, we find that steps in [10, 20] can generate quite good samples.
## And steps = 20 can almost converge.
x_sample = dpm_solver.sample(
    x_T,
    steps=15,
    order=2,
    skip_type="time_uniform",
    method="multistep",
)
```

<br />

# Documentation

Coming soon...

<!-- 
## 1. Define the noise schedule.
We support the 'linear' or 'cosine' VP noise schedule. For example, for the commly-used linear schedule (i.e. the $\beta_t$ is a linear function of $t$, as used in [DDPM](https://arxiv.org/abs/2006.11239)), you need to define:
```python
from dpm_solver_pytorch import NoiseScheduleVP

noise_schedule = NoiseScheduleVP(schedule='linear')
```

If you want to custom your own designed noise schedule, you need to implement the `marginal_log_mean_coeff`, `marginal_std`, `marginal_lambda` and `inverse_lambda` functions of your noise schedule. Please refer to the detailed comments in the code of `NoiseScheduleVP`.

<br />

## 2. Wrap your noise prediction model to the continuous-time model.

For a given noise prediction model (i.e. the $\epsilon_{\theta}(x_t, t)$ ) `model` and `model_kwargs` with the following format:

```python
model(x_t, t_input, **model_kwargs)
```

where `t_input` is the time label of the model.
(may be discrete-time labels (i.e. 0 to 999) or continuous-time labels (i.e. 0 to $T$).)

We wrap the model function to the following format:
```python
model_fn(x, t_continuous)
```

where `t_continuous` is the continuous time labels (i.e. 0 to $T$). And we use `model_fn` for DPM-Solver.

Note that DPM-Solver only needs the noise prediction model (the "mean" model), so for diffusion models which predict both "mean" and "variance" (such as [improved-DDPM](https://arxiv.org/abs/2102.09672)), you need to firstly define another function by your model to only output the "mean".

If you want to custom your own designed model time input `t_input`, you need to modify the function `get_model_input_time` in the function `model_wrapper` to add a new time input type.

<br />

### 2.1. Continuous-time DPMs
For continuous-time DPMs, we have `t_input = t_continuous`. You can let `time_input_type` be `"0"` to wrap the model function:
```python
from dpm_solver_pytorch import model_wrapper

model_fn = model_wrapper(
    model,
    noise_schedule,
    is_cond_classifier=False,
    time_input_type="0",
    model_kwargs=model_kwargs
)
```

<br />

### 2.2. Discrete-time DPMs
For discrete-time DPMs, we support two types for converting the discrete time to the continuous time (see Appendix in our paper). We recommend `time_input_type = "1"` (the default setting). You also need to specify the total length of the discrete time (default is `1000`):
```python
from dpm_solver_pytorch import model_wrapper

model_fn = model_wrapper(
    model,
    noise_schedule,
    is_cond_classifier=False,
    time_input_type="1",
    total_N=1000,
    model_kwargs=model_kwargs
)
```

<br />

### 2.3. DPMs with classifier guidance
For DPMs with classifier guidance, we also combine the model output with the classifier gradient. You need to specify the classifier function and the guidance scale. The classifier function has the following format:
```python
classifier_fn(x_t, t_input)
```
where `t_input` is the same time label as in the original diffusion model `model`. For example, for discrete-time DPMs with classifier guidance:
```python
from dpm_solver_pytorch import model_wrapper

model_fn = model_wrapper(
    model,
    noise_schedule,
    is_cond_classifier=True,
    classifier_fn=classifier_fn,
    classifier_scale=1.,
    time_input_type="1",
    total_N=1000,
    model_kwargs=model_kwargs
)
```

<br />

## 3. Define DPM-Solver and compute samples
Just let
```python
from dpm_solver_pytorch import DPM_Solver

dpm_solver = DPM_Solver(model_fn, noise_schedule)
```
where `model_fn` is the output of the function `model_wrapper`.

You can use `dpm_solver.sample` to quickly sample from DPMs. This function computes the sample at time `eps` by DPM-Solver, given the initial `x_T` at time `T`.

We support the following algorithms:
- (**Recommended**) Fast version of DPM-Solver (i.e. DPM-Solver-fast), which uses uniform logSNR steps and combine different orders of DPM-Solver. 

- Adaptive step size DPM-Solver (i.e. DPM-Solver-12 and DPM-Solver-23)

- Fixed order DPM-Solver (i.e. DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3).

**We recommend DPM-Solver-fast for both fast sampling in few steps (<=20) and fast convergence in many steps (converges in 50 steps).**

<br />

### 3.1. (Recommended) Sampling by DPM-Solver-fast
Let `adaptive_step_size=False` and `fast_version=True`.

We recommend `eps=1e-3` for `steps <= 15`, and `eps=1e-4` for `steps > 15`. For example:

* If you want to get a fine sample as fast as possible, you can use `eps=1e-3` and `steps=12`.

* If you want to get a quite good sample, you can use `eps=1e-4` and `steps=20`.

* If you want to get a best (converged) sample, you can use `eps=1e-4` and `steps=50`.


For example, to sample with NFE = 15:
```python
x_sample = dpm_solver.sample(
    x_T,
    steps=15,
    eps=1e-4,
    adaptive_step_size=False,
    fast_version=True,
)
```

More precisely, given a fixed NFE=`steps`, the sampling procedure by DPM-Solver-fast is:

- Denote `K = (steps // 3 + 1)`. We take `K` intermediate time steps for sampling.

- If `steps % 3 == 0`, we use `K - 2` steps of DPM-Solver-3, and `1` step of DPM-Solver-2 and 1 step of DPM-Solver-1.

- If `steps % 3 == 1`, we use `K - 1` steps of DPM-Solver-3 and `1` step of DPM-Solver-1.

- If `steps % 3 == 2`, we use `K - 1` steps of DPM-Solver-3 and `1` step of DPM-Solver-2.


<br />

### 3.2. Sampling by adaptive step size DPM-Solver
Let `adaptive_step_size=True`.

We recommend `eps=1e-4` for better sample quality.

If `order`=2, we use DPM-Solver-12 which combines DPM-Solver-1 and DPM-Solver-2.

If `order`=3, we use DPM-Solver-23 which combines DPM-Solver-2 and DPM-Solver-3.

You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs (NFE) and the sample quality. For image data, we recommend `atol=0.0078` (the default setting).

For example, to sample by DPM-Solver-12:
```python
x_sample = dpm_solver.sample(
    x_T,
    eps=1e-4,
    order=2,
    adaptive_step_size=True,
    fast_version=False,
    rtol=0.05,
)
```

<br />

### 3.3. Sampling by DPM-Solver-k for k = 1, 2, 3
Let `adaptive_step_size=False` and `fast_version=False`.

We use DPM-Solver-`order` for `order` = 1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.

We support three types of `skip_type`:

- 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.

- 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)

- 'time_quadratic': quadratic time for the time steps. (Used in DDIM.)

For example, to sample by DPM-Solver-3:
```python
x_sample = dpm_solver.sample(
    x_T,
    steps=30,
    eps=1e-4,
    order=3,
    skip_type='logSNR',
    adaptive_step_size=False,
    fast_version=False,
)
```

<br /> -->

# Examples
We also add a pytorch example and a JAX example. The documentations are coming soon.

<br />

# TODO List
- [ ] Add stable-diffusion examples.
- [ ] Documentation for example code.
- [ ] Clean and add the JAX code example.
- [ ] Add more explanations about DPM-Solver.
- [ ] Add VE type noise schedule.



<br />

# References

If you find the code useful for your research, please consider citing
```bib
@article{lu2022dpm,
  title={DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps},
  author={Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2206.00927},
  year={2022}
}
```

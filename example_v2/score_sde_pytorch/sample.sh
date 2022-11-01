devices="0"

steps="10"
eps="1e-3"
skip="logSNR"
method="singlestep"
order="3"
dir="experiments/cifar10_ddpmpp_deep_continuous_steps"

CUDA_VISIBLE_DEVICES=$devices python main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --mode "eval" --workdir $dir --config.sampling.eps=$eps --config.sampling.method="dpm_solver" --config.sampling.steps=$steps --config.sampling.skip_type=$skip --config.sampling.dpm_solver_order=$order --config.sampling.dpm_solver_method=$method --config.eval.batch_size=1000

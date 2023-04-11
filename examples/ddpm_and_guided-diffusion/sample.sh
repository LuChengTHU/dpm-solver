DEVICES='4,5'


##########################

# CIFAR-10 (DDPM checkpoint) example

data="cifar10"
sampleMethod='dpmsolver++'
type="dpmsolver"
steps="10"
DIS="logSNR"
order="3"
method="multistep"
workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 


#########################

# ImageNet64 (improved-DDPM checkpoint) example

data="imagenet64"
sampleMethod='dpmsolver++'
type="dpmsolver"
steps="10"
DIS="logSNR"
order="3"
method="multistep"
workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 


#########################

# ImageNet256 with classifier guidance (large guidance scale) example

data="imagenet256_guided"
scale="8.0"
sampleMethod='dpmsolver++'
type="dpmsolver"
steps="20"
DIS="time_uniform"
order="2"
method="multistep"

workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_scale"$scale"_type-"$type"_thresholding"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 --scale=$scale --thresholding

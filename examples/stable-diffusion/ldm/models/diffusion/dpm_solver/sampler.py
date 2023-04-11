"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))
        self.noise_schedule = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               skip_type="time_uniform",
               method="multistep",
               order=2,
               lower_order_final=True,
               correcting_xt_fn=None,
               t_start=None,
               t_end=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            self.noise_schedule,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, self.noise_schedule, algorithm_type="dpmsolver++", correcting_xt_fn=correcting_xt_fn)

        x, intermediates = dpm_solver.sample(img, t_start=t_start, t_end=t_end, steps=S, skip_type=skip_type, method=method, order=order, lower_order_final=lower_order_final, return_intermediate=True)

        return x.to(device), intermediates

    @torch.no_grad()
    def stochastic_encode(self, x0, encode_ratio, noise=None):
        t_end = self.ratio_to_time(encode_ratio)
        t_end = torch.tensor([t_end], device=x0.device, dtype=x0.dtype)
        x = DPM_Solver(None, self.noise_schedule).add_noise(x0, t_end, noise=noise)
        return x

    @torch.no_grad()
    def encode(self,
               S,
               x,
               encode_ratio,
               conditioning=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               skip_type="time_uniform",
               method="multistep",
               order=2,
               lower_order_final=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != x.shape[0]:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {x.shape[0]}")
            else:
                if conditioning.shape[0] != x.shape[0]:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {x.shape[0]}")

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            self.noise_schedule,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        t_end = self.ratio_to_time(encode_ratio)

        dpm_solver = DPM_Solver(model_fn, self.noise_schedule, algorithm_type="dpmsolver++")

        x, intermediates = dpm_solver.inverse(x, steps=S, t_end=t_end, skip_type=skip_type, method=method, order=order, lower_order_final=lower_order_final, return_intermediate=True)

        return x, intermediates

    def time_discrete_to_continuous(self, t_discrete):
        """
        Convert [0, 999] to [0.001, 1].
        """
        return (t_discrete + 1.) / self.noise_schedule.total_N

    def time_continuous_to_discrete(self, t_continuous):
        """
        Convert [0.001, 1] to [0, 999].
        """
        return t_continuous * self.noise_schedule.total_N - 1. 

    def ratio_to_time(self, ratio):
        """
        Convert [0, 1] to [0.001, 1].
        """
        return (1. - 1. / self.noise_schedule.total_N) * ratio + 1. / self.noise_schedule.total_N

    def time_to_ratio(self, t_continuous):
        """
        Convert [0.001, 1] to [0, 1].
        """
        return (t_continuous - 1. / self.noise_schedule.total_N) / (1. - self.noise_schedule.total_N)

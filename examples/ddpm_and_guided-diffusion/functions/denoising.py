import torch
import torch.nn.functional as F


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def cond_fn(x, t_discrete, y, classifier, classifier_scale):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t_discrete)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


def generalized_steps(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    device = x.device
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    device = x.device
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(device)

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds

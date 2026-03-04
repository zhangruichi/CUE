import torch
import json

def vlm_multi_fn(zs_model, imgs, labels, topk=5):
    with torch.no_grad():
        logits = zs_model(imgs)
        zs_logit = logits.clone()
        zs_logit[torch.arange(len(labels)), labels] = -float("inf")
        topk_indices = torch.topk(zs_logit, topk, dim=-1).indices
        num_classes = logits.size(1)
        multi_targets = torch.zeros((imgs.size(0), num_classes), device=imgs.device)
        multi_targets.scatter_(1, labels.unsqueeze(1), 1.0)
        multi_targets.scatter_(1, topk_indices, 1.0)
        multi_targets = multi_targets.clamp_(min=0.0, max=1.0)
    return multi_targets


def get_multi_fn(classes, json_path, device):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    C = len(classes)
    base = torch.zeros((C, C))
    for i, name in enumerate(classes):
        for j, cname in enumerate(classes):
            if cname == name:
                base[i, j] = 1

        for s in data.get(name, []):
            js = [j for j, cname in enumerate(classes) if cname == s]
            for j in js:
                base[i, j] = 1

    base = base.to(device)

    def fn(labels):
        return base[labels]

    return fn



def get_multi_bla_loss(cls_num_list, tau=1.0):
    cls_num_list = torch.tensor(cls_num_list)
    cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
    log_cls_num = torch.log(cls_num_ratio)
    tau = torch.tensor(tau)

    def loss_fn(logit, target):
        logit_adjusted = logit + tau * log_cls_num.unsqueeze(0).to(logit.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(logit_adjusted, target)

    return loss_fn

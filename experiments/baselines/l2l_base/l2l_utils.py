import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score
from torch.func import functional_call


def func_call(model, params_dict, args):
    if params_dict is None:
        params_dict = dict(model.named_parameters())
    y = functional_call(model, params_dict, args)
    return y


def zeroed_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    return


def count_parameters(model):
    nweights = []
    for name, param in model.named_parameters():
        if any(
            word in name.split(".") for word in model.pred_with_transformer
        ):
            nweights.append(param.numel())
    return sum(nweights)


class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, config_params, task_id):
        self.warmup_steps = config_params["warmup_steps"]
        self.total_steps = config_params["steps"]
        self.lr_init = config_params["lr"]
        self.task_id = task_id
        super().__init__(optimizer)

    def get_lr(self):
        if self.task_id == 0:
            if self.last_epoch < self.warmup_steps:
                lr = self.lr_init * (self.last_epoch / self.warmup_steps)
            else:
                lr = self.lr_init * max(
                    0.0,
                    float(self.total_steps - self.last_epoch)
                    / float(max(1, self.total_steps - self.warmup_steps)),
                )
        else:
            lr = self.lr_init * max(
                0.0,
                float(self.total_steps - self.last_epoch)
                / float(max(1, self.total_steps)),
            )
        return [lr for _ in self.optimizer.param_groups]


def set_optimizer(model, transformer_model, config_params):
    parameters_model = {
        n: w
        for n, w in model.named_parameters()
        if not any(
            word in n.split(".") for word in model.pred_with_transformer
        )
    }
    parameters_transformer = transformer_model.parameters()

    model_optimizer = torch.optim.Adam(
        parameters_model.values(),
        lr=config_params["lr"],
        weight_decay=config_params["lambda_l2"],
    )
    transformer_optimizer = torch.optim.Adam(
        parameters_transformer,
        lr=config_params["lr"],
        weight_decay=config_params["lambda_l2"],
    )

    return model_optimizer, transformer_optimizer


def l2_regularization(dict_parameters):
    l2reg_loss = 0.0
    for param in dict_parameters.values():
        l2reg_loss += torch.norm(param, p=2)
    return l2reg_loss


def select_top_k(importance_scores, K):
    new_importance_scores = importance_scores.copy()
    for name, scores in importance_scores.items():
        flattened_scores = torch.cat([i.view(-1) for i in scores])
        sorted_scores, _ = torch.sort(flattened_scores, descending=True)

        threshold_index = int(len(sorted_scores) * K)
        threshold_value = sorted_scores[threshold_index]

        mask = flattened_scores >= threshold_value
        mask = mask.view(scores.shape)

        new_importance_scores[name] = scores * mask
    return new_importance_scores


def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)

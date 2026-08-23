import torch
import torch.nn as nn
import torchvision.models as tv_models

from utils import _name_matches_any_token


def get_all_param_owning_modules(model):
    """All non-root modules that own at least one direct parameter."""
    result = []
    for name, module in model.named_modules():
        if not name:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            result.append(name)
    return result


def resolve_training_layer_tokens(model, training_layers):
    """Turns a `training_layers` spec into the module-name tokens assigned to
    `model.pred_with_transformer`.

    None or "all": every parameter-owning module. "all_but_head": every
    feature layer with a non-constant init; the head ("last") and
    constant-init modules (BatchNorm/LayerNorm, biases) stay on Adam, as in
    the base paper. A list of names is validated and passed through.
    """
    if training_layers is None or training_layers == "all":
        return get_all_param_owning_modules(model)

    if training_layers == "all_but_head":
        # The std test mirrors the init_std floor PROPOSED uses to pick
        # managed tensors, so this token set equals the managed set after it.
        managed, skipped_head, skipped_const = [], [], []
        for name, module in model.named_modules():
            if not name:
                continue
            owns_params = any(True for _ in module.parameters(recurse=False))
            if "last" in name.split("."):
                if owns_params:
                    skipped_head.append(name)
                continue
            if not owns_params:
                continue
            params = list(module.parameters(recurse=False))
            if any(
                p.numel() > 1 and float(p.detach().std()) > 1e-6 for p in params
            ):
                managed.append(name)
            else:
                skipped_const.append(name)
        print(
            f"[all_but_head] managed ({len(managed)}): {managed}\n"
            f"[all_but_head] skipped head ({len(skipped_head)}): {skipped_head}\n"
            f"[all_but_head] skipped constant-init ({len(skipped_const)}): "
            f"{skipped_const}"
        )
        return managed

    if not isinstance(training_layers, (list, tuple)):
        raise TypeError(
            f"training_layers must be None, 'all', or a list of strings. "
            f"Got {type(training_layers).__name__}."
        )

    all_parameter_names = [name for name, _ in model.named_parameters()]
    unmatched = [
        token
        for token in training_layers
        if not any(
            _name_matches_any_token(name, [token]) for name in all_parameter_names
        )
    ]
    if unmatched:
        valid_modules = get_all_param_owning_modules(model)
        raise ValueError(
            f"training_layers contains names that do not match any parameter "
            f"in {type(model).__name__}: {unmatched}. "
            f"Valid module names: {valid_modules}"
        )
    return list(training_layers)


class ConvNet(nn.Module):
    def __init__(self, config_params, n_heads=None, n_classes=None):
        super().__init__()
        self.input_dim = config_params["input_dim"]
        self.features_dim = config_params["features_dim"]
        self.num_classes = n_classes
        self.config_params = config_params
        self.n_heads = n_heads

        self.conv1 = nn.Conv2d(
            self.input_dim, self.features_dim, 3, padding="same", bias=False
        )
        if self.config_params["use_bn"]:
            self.bn1 = nn.BatchNorm2d(
                self.features_dim, track_running_stats=False
            )
        self.act = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()

        dim_flattened = int(
            self.features_dim
            * int(self.config_params["input_size"] / 2)
            * int(self.config_params["input_size"] / 2)
        )
        self.last = (
            nn.Linear(dim_flattened, self.num_classes, bias=False)
            if n_heads is None
            else nn.ModuleList(
                [
                    nn.Linear(dim_flattened, self.num_classes, bias=False)
                    for _ in range(n_heads)
                ]
            )
        )

        self.initialize_weights()

        # Default scope; apply_pred_strategy overwrites it from config.
        self.pred_with_transformer = ["conv1"]
        if config_params.get("predict_bn", False):
            self.pred_with_transformer.append("bn1")

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, task_idx=None):
        x = self.conv1(x)
        if self.config_params["use_bn"]:
            x = self.bn1(x)
        x = self.maxpool(self.act(x))
        x = self.flat(x)
        out = self.last(x) if self.n_heads is None else self.last[task_idx](x)
        return out


class ThreeConvNetSimple(nn.Module):
    def __init__(self, config_params, n_heads=None, n_classes=None):
        super().__init__()
        self.input_dim = config_params["input_dim"]
        self.features_dim = config_params["features_dim"]
        self.num_classes = n_classes
        self.config_params = config_params
        self.n_heads = n_heads

        self.conv1 = nn.Conv2d(self.input_dim, self.features_dim, 3, bias=False)
        self.conv2 = nn.Conv2d(self.features_dim, self.features_dim, 3, bias=False)
        self.conv3 = nn.Conv2d(self.features_dim, self.features_dim, 3, bias=False)
        if self.config_params["use_bn"]:
            self.bn1 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)
            self.bn2 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)
            self.bn3 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)

        self.act = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        dim_flatten = (
            self.features_dim
            * int(self.config_params["input_size"] / 16)
            * int(self.config_params["input_size"] / 16)
        )
        self.last = (
            nn.Linear(dim_flatten, self.num_classes, bias=False)
            if n_heads is None
            else nn.ModuleList(
                [
                    nn.Linear(dim_flatten, self.num_classes, bias=False)
                    for _ in range(n_heads)
                ]
            )
        )

        self.initialize_weights()

        self.pred_with_transformer = ["conv3"]
        if config_params.get("predict_bn", False):
            self.pred_with_transformer.append("bn3")

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, task_idx=None):
        x = self.conv1(x)
        if self.config_params["use_bn"]:
            x = self.bn1(x)
        x = self.maxpool(self.act(x))
        x = self.conv2(x)
        if self.config_params["use_bn"]:
            x = self.bn2(x)
        x = self.maxpool(self.act(x))
        x = self.conv3(x)
        if self.config_params["use_bn"]:
            x = self.bn3(x)
        x = self.maxpool(self.act(x))
        x = self.flat(x)
        out = self.last(x) if self.n_heads is None else self.last[task_idx](x)
        return out


class ResNet18(nn.Module):
    def __init__(self, config_params, n_heads=None, n_classes=None):
        super().__init__()
        self.config_params = config_params
        self.n_heads = n_heads

        resnet = tv_models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(
            config_params["input_dim"], 64, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        resnet.maxpool = nn.Identity()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.flat = nn.Flatten()

        self.last = (
            nn.Linear(512, n_classes, bias=False)
            if n_heads is None
            else nn.ModuleList(
                [nn.Linear(512, n_classes, bias=False) for _ in range(n_heads)]
            )
        )

        self.pred_with_transformer = ["layer4"]

    def forward(self, x, task_idx=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flat(x)
        out = self.last(x) if self.n_heads is None else self.last[task_idx](x)
        return out


class ViTSmall(nn.Module):
    def __init__(self, config_params, n_heads=None, n_classes=None):
        super().__init__()
        self.config_params = config_params
        self.n_heads = n_heads

        patch_size = config_params.get("vit_patch_size", 4)
        img_size = config_params["input_size"]
        in_channels = config_params["input_dim"]
        embed_dim = config_params["features_dim"]
        nhead = config_params.get("vit_num_heads", 6)
        num_layers = config_params.get("vit_num_layers", 6)
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.last = (
            nn.Linear(embed_dim, n_classes, bias=False)
            if n_heads is None
            else nn.ModuleList(
                [nn.Linear(embed_dim, n_classes, bias=False) for _ in range(n_heads)]
            )
        )

        self.pred_with_transformer = ["encoder"]

    def forward(self, x, task_idx=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        out = self.last(x) if self.n_heads is None else self.last[task_idx](x)
        return out

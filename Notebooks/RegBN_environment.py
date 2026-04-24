#This is a supplimental file to RegBN_Implementation.ipynb. Features from NNpipeline are refactored to work with RegBN in the main notebook.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from RegBN import RegBN


NUMERIC_COLS = [
    "deflated_gdp_usd",
    "us_cpi",
    "straight_distance_to_capital_km",
    "quarter_label",
]

CATEGORICAL_COLS = [
    "geolocation_name",
    "country",
    "landlocked",
    "region_economic_classification",
    "access_to_airport",
    "access_to_port",
    "access_to_highway",
    "access_to_railway",
    "seismic_hazard_zone",
    "flood_risk_class",
    "tropical_cyclone_wind_risk",
    "tornadoes_wind_risk",
    "koppen_climate_zone",
]

TARGET_COL = "construction_cost_per_m2_usd"
IMG_COL = "processed_imgs"
ID_COL = "data_id"


def load_tensor_dict(img_dir: Path) -> dict[str, torch.Tensor]:
    tensor_dict: dict[str, torch.Tensor] = {}
    for img_file in sorted(img_dir.glob("*.pt")):
        img_dict = torch.load(img_file, weights_only=True)
        tensor_dict[img_file.name] = torch.cat([img_dict["sentinel"], img_dict["viirs"]], dim=0)
    return tensor_dict


class ConstructionDataset(Dataset):
    """Lazy multimodal dataset backed by the processed CSV outputs."""

    def __init__(self, df: pd.DataFrame, tensor_dict: dict[str, torch.Tensor], augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.tensor_dict = tensor_dict
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        continuous_cols = [col for col in NUMERIC_COLS if col in self.df.columns]
        categorical_cols = [col for col in CATEGORICAL_COLS if col in self.df.columns]

        data_id = row[ID_COL]
        continuous = torch.tensor(row[continuous_cols].values.astype(np.float32), dtype=torch.float32)
        categoricals = {
            name: torch.tensor(int(row[name]), dtype=torch.long)
            for name in categorical_cols
        }
        images = self.tensor_dict[row[IMG_COL]]
        if TARGET_COL in self.df.columns:
            target = torch.tensor(float(row[TARGET_COL]), dtype=torch.float32)
        else:
            target = torch.tensor(np.nan, dtype=torch.float32)

        if self.augment:
            if torch.rand(1) > 0.5:
                images = torch.flip(images, dims=[2])
            if torch.rand(1) > 0.5:
                images = torch.flip(images, dims=[1])

        return data_id, continuous, categoricals, images, target


def collate_fn(batch):
    data_ids = [item[0] for item in batch]
    continuous = default_collate([item[1] for item in batch])
    categoricals = default_collate([item[2] for item in batch])
    images = default_collate([item[3] for item in batch])
    targets = default_collate([item[4] for item in batch])
    return data_ids, continuous, categoricals, images, targets


def interpolate_datapoints(
    a: pd.Series,
    b: pd.Series,
    quarter_label: float,
    tensor_dict: dict[str, torch.Tensor],
    lam: float,
):
    """Interpolate numeric, categorical, tensor, and target values between two rows."""
    c = pd.Series(dtype=object)
    keys = a.keys()
    numeric_cols = [col for col in NUMERIC_COLS if col in keys]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in keys]

    for col in numeric_cols:
        c[col] = quarter_label if col == "quarter_label" else lam * a[col] + (1 - lam) * b[col]

    for col in categorical_cols:
        c[col] = a[col] if lam >= 0.5 else b[col]

    c_path = "Interpolation/" + a[IMG_COL][:-3] + "_" + b[IMG_COL][:-3] + ".pt"
    c[IMG_COL] = c_path
    tensor_dict[c_path] = torch.lerp(tensor_dict[b[IMG_COL]], tensor_dict[a[IMG_COL]], lam)

    c[TARGET_COL] = lam * a[TARGET_COL] + (1 - lam) * b[TARGET_COL]
    c[ID_COL] = f"{a[ID_COL]}_{b[ID_COL]}_{quarter_label:.4f}"

    for col in keys:
        if col not in c.index:
            c[col] = a[col] if col in categorical_cols else b[col]

    return c, tensor_dict


def geolocation_temporal_interpolation(
    df: pd.DataFrame,
    tensor_dict: dict[str, torch.Tensor],
    max_missing: Optional[int] = None,
):
    """Reproduce the temporal interpolation used in the current NN pipeline."""
    df = df.copy()
    tensor_dict = dict(tensor_dict)
    quarter_labels = sorted(df["quarter_label"].unique())
    geolocations = sorted(df["geolocation_name"].unique())

    for geo in geolocations:
        geo_df = df[df["geolocation_name"] == geo]
        geo_quarters = set(geo_df["quarter_label"].values)
        last_quarter = None
        missing_quarters: list[float] = []
        row = None
        for quarter in quarter_labels:
            if quarter in geo_quarters:
                new_row = geo_df[geo_df["quarter_label"] == quarter].iloc[0]
                interpolations = len(missing_quarters)
                if last_quarter is not None and (max_missing is None or interpolations <= max_missing):
                    delta_lambda = 1 / (1 + interpolations)
                    synthetic_rows = []
                    for i in range(interpolations):
                        lam = 1 - delta_lambda * (1 + i)
                        synth_row, tensor_dict = interpolate_datapoints(
                            row,
                            new_row,
                            missing_quarters[i],
                            tensor_dict,
                            lam,
                        )
                        synthetic_rows.append(synth_row)
                    if synthetic_rows:
                        df_synthetic = pd.DataFrame(synthetic_rows, columns=df.columns)
                        df = pd.concat([df, df_synthetic], ignore_index=True)
                missing_quarters = []
                row = new_row
                last_quarter = quarter
            elif last_quarter is not None:
                missing_quarters.append(quarter)

    return df, tensor_dict


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularEncoder(nn.Module):
    """Embedding + MLP encoder for the processed economic features."""

    def __init__(
        self,
        continuous_dim: int,
        categorical_vocab: dict[str, int],
        embed_dim_fn=lambda v: min(50, (v + 1) // 2),
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size + 1, embed_dim_fn(vocab_size))
                for name, vocab_size in categorical_vocab.items()
            }
        )

        total_embed_dim = sum(embed_dim_fn(v) for v in categorical_vocab.values())
        mlp_in = continuous_dim + total_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, continuous: torch.Tensor, categoricals: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [continuous]
        for name, emb in self.embeddings.items():
            parts.append(emb(categoricals[name]))
        return self.mlp(torch.cat(parts, dim=1))


class ImageEncoder(nn.Module):
    """Compact CNN for the 13-channel Sentinel-2 + VIIRS tensors."""

    def __init__(
        self,
        in_channels: int = 13,
        base_ch: int = 32,
        out_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            ResBlock(base_ch),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),
            ResBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),
            ResBlock(base_ch * 2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(),
            ResBlock(base_ch * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.stage1(imgs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class GatedFusion(nn.Module):
    def __init__(self, tab_dim: int = 128, img_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.tab_proj = nn.Linear(tab_dim, out_dim)
        self.img_proj = nn.Linear(img_dim, out_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(tab_dim + img_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, tab: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(torch.cat([tab, img], dim=1))
        return gate * self.tab_proj(tab) + (1 - gate) * self.img_proj(img)


@dataclass
class RegBNContext:
    is_training: bool
    n_epoch: int = 0
    steps_per_epoch: int = 1


class MultimodalRegBN(nn.Module):
    """
    Apply RegBN between modality embeddings before fusion.

    Modes:
    - image: debias image embeddings with respect to tabular embeddings.
    - tabular: debias tabular embeddings with respect to image embeddings.
    - bidirectional: run both directions and fuse the cleaned outputs.
    """

    def __init__(
        self,
        tab_dim: int,
        img_dim: int,
        device: torch.device,
        mode: str = "bidirectional",
        normalize_input: bool = True,
        normalize_output: bool = True,
        sigma_thr: float = 0.0,
        sigma_min: float = 0.0,
        momentum: float = 0.02,
        verbose: bool = False,
        projection_clip_value: float = 10.0,
    ):
        super().__init__()
        valid_modes = {"image", "tabular", "bidirectional"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

        self.mode = mode
        self.projection_clip_value = projection_clip_value
        self.image_from_tab = RegBN(
            f_num_channels=img_dim,
            g_num_channels=tab_dim,
            f_layer_dim=[],
            g_layer_dim=[],
            device=device,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            affine=False,
            sigma_THR=sigma_thr,
            sigma_MIN=sigma_min,
            momentum=momentum,
            verbose=verbose,
        )
        self.tab_from_image = RegBN(
            f_num_channels=tab_dim,
            g_num_channels=img_dim,
            f_layer_dim=[],
            g_layer_dim=[],
            device=device,
            normalize_input=normalize_input,
            normalize_output=normalize_output,
            affine=False,
            sigma_THR=sigma_thr,
            sigma_MIN=sigma_min,
            momentum=momentum,
            verbose=verbose,
        )

    def _sanitize_projection(self, module: RegBN) -> torch.Tensor:
        W = torch.nan_to_num(module.W.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.projection_clip_value is not None:
            W = W.clamp(-self.projection_clip_value, self.projection_clip_value)
        if not torch.isfinite(W).all():
            W = torch.zeros_like(W)
        return W

    def _apply_projection(self, module: RegBN, f: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.isfinite(f).all() or not torch.isfinite(g).all():
            return f, g

        W = self._sanitize_projection(module)
        f_sz, g_sz = f.size(), g.size()

        f_flat = f.reshape(f_sz[0], -1)
        g_flat = g.reshape(g_sz[0], -1)
        f_mapped2g = torch.mm(g_flat, W)
        f_residual = (f_flat - f_mapped2g).reshape(f_sz)

        f_residual = module.norm_f_out(f_residual)
        g_out = module.norm_g_out(g)

        f_residual = torch.nan_to_num(f_residual, nan=0.0, posinf=0.0, neginf=0.0)
        g_out = torch.nan_to_num(g_out, nan=0.0, posinf=0.0, neginf=0.0)
        return f_residual, g_out

    def _safe_regbn_pass(
        self,
        module: RegBN,
        f: torch.Tensor,
        g: torch.Tensor,
        context: RegBNContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if f.shape[0] < 2 or g.shape[0] < 2:
            return f, g

        if context.is_training and torch.isfinite(f).all() and torch.isfinite(g).all():
            try:
                module(
                    f.detach(),
                    g.detach(),
                    is_training=True,
                    n_epoch=context.n_epoch,
                    steps_per_epoch=max(context.steps_per_epoch, 1),
                )
            except RuntimeError:
                module.W.zero_()
                module.is_nan_ = False

        try:
            f_clean, g_clean = self._apply_projection(module, f, g)
        except RuntimeError:
            module.W.zero_()
            module.is_nan_ = False
            return f, g

        if not torch.isfinite(f_clean).all() or not torch.isfinite(g_clean).all():
            module.W.zero_()
            module.is_nan_ = False
            return f, g
        return f_clean, g_clean

    def forward(
        self,
        tab: torch.Tensor,
        img: torch.Tensor,
        context: Optional[RegBNContext] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None:
            context = RegBNContext(is_training=self.training)

        if self.mode == "image":
            img_clean, _ = self._safe_regbn_pass(self.image_from_tab, img, tab, context)
            return tab, img_clean

        if self.mode == "tabular":
            tab_clean, _ = self._safe_regbn_pass(self.tab_from_image, tab, img, context)
            return tab_clean, img

        tab_clean, _ = self._safe_regbn_pass(self.tab_from_image, tab, img, context)
        img_clean, _ = self._safe_regbn_pass(self.image_from_tab, img, tab, context)
        return tab_clean, img_clean


class ConstructionCostRegBNModel(nn.Module):
    """Baseline multimodal model with an optional RegBN block before fusion."""

    def __init__(
        self,
        df: pd.DataFrame,
        device: torch.device,
        tab_out_dim: int = 128,
        img_out_dim: int = 64,
        fusion_dim: int = 128,
        head_hidden: int = 64,
        dropout: float = 0.3,
        img_channels: int = 13,
        regbn_mode: Optional[str] = "bidirectional",
        regbn_normalize_input: bool = True,
        regbn_normalize_output: bool = True,
        regbn_sigma_thr: float = 0.0,
        regbn_sigma_min: float = 0.0,
        regbn_momentum: float = 0.02,
        regbn_verbose: bool = False,
        regbn_projection_clip_value: float = 10.0,
    ):
        super().__init__()

        cols = df.columns
        continuous_dim = sum(1 for col in NUMERIC_COLS if col in cols)
        categorical_vocab = {
            col: int(df[col].max())
            for col in CATEGORICAL_COLS
            if col in cols
        }

        self.tabular_encoder = TabularEncoder(
            continuous_dim=continuous_dim,
            categorical_vocab=categorical_vocab,
            out_dim=tab_out_dim,
            dropout=dropout,
        )
        self.image_encoder = ImageEncoder(
            in_channels=img_channels,
            out_dim=img_out_dim,
            dropout=dropout,
        )
        self.regbn = None
        if regbn_mode is not None:
            self.regbn = MultimodalRegBN(
                tab_dim=tab_out_dim,
                img_dim=img_out_dim,
                device=device,
                mode=regbn_mode,
                normalize_input=regbn_normalize_input,
                normalize_output=regbn_normalize_output,
                sigma_thr=regbn_sigma_thr,
                sigma_min=regbn_sigma_min,
                momentum=regbn_momentum,
                verbose=regbn_verbose,
                projection_clip_value=regbn_projection_clip_value,
            )
        self.fusion = GatedFusion(
            tab_dim=tab_out_dim,
            img_dim=img_out_dim,
            out_dim=fusion_dim,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        continuous: torch.Tensor,
        categoricals: dict[str, torch.Tensor],
        images: torch.Tensor,
        regbn_context: Optional[RegBNContext] = None,
    ) -> torch.Tensor:
        tab = self.tabular_encoder(continuous, categoricals)
        img = self.image_encoder(images)
        if self.regbn is not None:
            tab, img = self.regbn(tab, img, context=regbn_context)
        fused = self.fusion(tab, img)
        return self.regression_head(fused).squeeze(1)


def rmsle_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.clamp(preds, min=0)
    return torch.sqrt(torch.mean((torch.log1p(preds) - torch.log1p(targets)) ** 2))

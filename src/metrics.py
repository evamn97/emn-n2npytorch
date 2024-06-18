#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union, List, Sequence, Optional, Tuple

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import mean_squared_error as mse_F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_F
from torchmetrics.functional.image.ssim import _ssim_update
from torchmetrics.functional.image.lpips import _NoTrainLpips
# import lpips
from typing import Literal, Any, Dict
from typing_extensions import override
from functools import partial
from numpy import ndarray
import pandas as pd


def stack_zero_dim(state_list: List[Tensor]) -> ndarray:
    """ Stacks a list of zero-dimensional tensors into a flattened numpy array.
        If input list is empty, return None. """
    
    if len(state_list) == 1:
        return state_list[0].detach().cpu().numpy()
    elif len(state_list) > 1:
        return torch.stack(state_list).detach().cpu().numpy()


class CustomFittingMetric(Metric):
    """ Custom base class for metric tracking over training and validation epochs. 
    
    ** Note that val=True and val=False method calls MUST be in different lightning hooks! 
    ** For example:

            def on_train_epoch_end(self):
                train_loss = self.loss.compute()
                valid_loss = self.loss.compute(val=True)
            
        will cause train_loss == valid_loss.
    
    ** Instead, try:

            def on_train_epoch_end(self):
                train_loss = self.loss.compute()
                ...
            
            def on_validation_epoch_end(self):
                valid_loss = self.loss.compute(val=True)
    """

    sum_metric: Tensor
    total: Tensor

    sum_metric_val: Tensor
    total_val: Tensor

    per_epoch: List
    per_epoch_val: List

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self, 
        data_range: Optional[Tuple[float, float]] = None, 
        **kwargs
        ) -> None:
        
        super().__init__(**kwargs)

        self.add_state("sum_metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("sum_metric_val", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_val", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # TODO: try changing to 'per_epoch = []' instead of using add_state so reset() doesn't clear it, and reset at end of each epoch
        # self.add_state("per_epoch", default=[], dist_reduce_fx="cat")
        # self.add_state("per_epoch_val", default=[], dist_reduce_fx="cat")
        self.per_epoch = []
        self.per_epoch_val = []

        if isinstance(data_range, tuple):
            self.add_state("data_range", default=torch.tensor(data_range[1] - data_range[0]), dist_reduce_fx="mean")
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
        else:
            self.data_range = None
            self.clamping_fn = None
    
    def calculate(self, 
                  *_: Any) -> Any:
        raise NotImplementedError
    
    @override
    def update(self, 
               preds: Tensor, 
               target: Tensor, 
               val: bool = False, 
               ) -> None:
        """ Update state with preds and target. """

        value = self.calculate(preds, target, reduction='sum')

        if val:
            self.sum_metric_val += value
            self.total_val += preds.shape[0]
        else:
            self.sum_metric += value
            self.total += preds.shape[0]
    
    @override
    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                val: bool = False, 
                ) -> Tensor:
        """ Calculate metric and update state. """

        self.update(preds, target, val)
        return self.calculate(preds, target)
    
    def update_on_epoch(self, 
                        val: bool = False
                        ) -> None:
        """ Update epoch-level state variables. """

        if val and self.total_val > 0:
            self.per_epoch_val.append((self.sum_metric_val / self.total_val).item())
        elif self.total > 0:
            self.per_epoch.append((self.sum_metric / self.total).item())

    @override
    def compute(self, 
                val: bool = False
                ) -> Tensor:
        """ Compute metric over state and update epoch tracker. """
        
        if val:
            return self.sum_metric_val / self.total_val
        else:
            return self.sum_metric / self.total


class MSE_SSIM_Loss(CustomFittingMetric):

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03, **kwargs
        ) -> None:

        super().__init__(data_range, **kwargs)

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2

    @property
    def name(self) -> str:
        return "mse-ssim"

    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor, 
                  reduction: Literal["mean", "sum"] = 'mean'
                  ) -> Tensor:
        """ Calculate metric from preds and target. """

        if len(preds.shape) not in (4, 5):
            raise ValueError(
                "Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape."
                f" Got preds: {preds.shape} and target: {target.shape}.")
        
        if self.data_range is None:
            self.data_range = (target.min().item(), target.max().item())
            self.clamping_fn = partial(torch.clamp, min=self.data_range[0], max=self.data_range[1])
        
        preds = self.clamping_fn(preds)
        target = self.clamping_fn(target)
        
        mse = torch.sum((preds - target).squeeze()**2, dim=-1).sum(dim=-1) / target.numel()

        similarity = _ssim_update(preds, target, 
                                  gaussian_kernel=self.gaussian_kernel, 
                                  sigma=self.sigma, 
                                  kernel_size=self.kernel_size, 
                                  data_range=self.data_range, 
                                  k1=self.k1, 
                                  k2=self.k2)
        
        combo = mse - similarity + 1
        if reduction == 'sum':
            return combo.sum()
        else: # reduction == 'mean':
            return combo.mean()


def F_MSE_SSIM(preds: Tensor, target: Tensor,
               gaussian_kernel: bool = True,
               sigma: Union[float, Sequence[float]] = 1.5,
               kernel_size: Union[int, Sequence[int]] = 11,
               data_range: Optional[Union[float, Tuple[float, float]]] = None,
               k1: float = 0.01,
               k2: float = 0.03):
    """ Functional method for calculating mse + (1 - ssim) directly """

    mse = mse_F(preds, target)
    ssim = ssim_F(preds, target,
                  gaussian_kernel=gaussian_kernel,
                  sigma=sigma,
                  kernel_size=kernel_size,
                  data_range=data_range,
                  k1=k1, k2=k2)

    return mse + (1 - ssim)


class LPIPS_Loss(CustomFittingMetric):
    """Averaging meter for tracking LPIPS loss over training and validation epochs. 
    Based on torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity class."""

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    
    def __init__(
        self, 
        net_type: Literal["vgg", "alex", "squeeze"] = 'vgg', 
        normalize: bool = True, 
        **kwargs
        ) -> None:

        super().__init__(**kwargs)

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}.")
        self.net = _NoTrainLpips(net=net_type)
        # self.net = lpips.LPIPS(net=net_type, verbose=False)

        if not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` should be an bool but got {normalize}")
        self.normalize = normalize

        if normalize:
            self.data_range = (0, 1)
        else:
            self.data_range = (-1, 1)
        self.clamping_fn = partial(torch.clamp, min=self.data_range[0], max=self.data_range[1])

    
    @property
    def name(self) -> str:
        return "lpips"

    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor, 
                  reduction: Literal["sum", "mean"] = 'mean'
                  ) -> Tensor:
        
        preds = self.clamping_fn(preds)
        target = self.clamping_fn(target)

        lpips = self.net.forward(preds.tile((1, 3, 1, 1)), 
                                 target.tile((1, 3, 1, 1)), 
                                 normalize=self.normalize)

        if reduction == 'sum':
            return lpips.sum()
        else:   # reduction == 'mean'
            return lpips.mean()


class PSNR_Meter(CustomFittingMetric):
    """Averaging meter for tracking PSNR over training and validation epochs. 
    Based on torchmetrics.image.psnr.PeakSignalNoiseRatio class."""

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs: Any,
        ) -> None:
        
        super().__init__(data_range, **kwargs)
    
    @property
    def name(self) -> str:
        return "psnr"

    def _mse(self, 
             preds: Tensor, 
             target: Tensor
             ) -> Tensor:
        # sets class-level data range for compute method on the first usage
        # !!! assumes data_range is the same for all targets !!!
        if self.data_range is None:
            self.data_range = (target.min().item(), target.max().item())
            self.clamping_fn = partial(torch.clamp, min=self.data_range[0], max=self.data_range[1])

        preds = self.clamping_fn(preds)
        target = self.clamping_fn(target)

        mse = torch.sum((preds - target).squeeze()**2, dim=-1).sum(dim=-1) / target.numel()
        return mse
    
    def _psnr(self, 
              mse: Tensor, 
              data_range: Union[float, Tensor], 
              ) -> Tensor:
        psnr = 10 * torch.log10(data_range**2 / mse)
        return psnr

    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor, 
                  reduction: Literal["mean", "sum"] = 'mean') -> Union[Tuple, Tensor]:
        """Calculate metric from preds and target. """

        mse = self._mse(preds, target)
        data_range = self.data_range[1] - self.data_range[0]

        psnr = self._psnr(mse, data_range)

        if reduction == 'sum':
            return psnr.sum()
        else: # reduction == 'mean'
            return psnr.mean()
    

class SSIM_Meter(CustomFittingMetric):
    """Averaging meter for tracking SSIM over training and validation epochs. 
    Based on torchmetrics.image.ssim.StructuralSimilarityIndexMeasure class."""

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False

    def __init__(
        self, 
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        **kwargs: Any,
        ) -> None:
        
        super().__init__(data_range, **kwargs)

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2

    @property
    def name(self) -> str:
        return "ssim"

    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor, 
                  reduction: Literal["mean", "sum"] = 'mean'
                  ) -> Tensor:
        
        if self.data_range is None:
            self.data_range = (target.min().item(), target.max().item())
            self.clamping_fn = partial(torch.clamp, min=self.data_range[0], max=self.data_range[1])
        
        preds = self.clamping_fn(preds)
        target = self.clamping_fn(target)

        similarity = _ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2)
        
        if reduction == 'sum':
            return similarity.sum()
        else: # reduction == 'mean':
            return similarity.mean()

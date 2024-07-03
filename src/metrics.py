#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union, List, Sequence, Optional, Tuple, Literal, Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.regression import mean_squared_error as mse_F
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as psnr_F
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as lpips_F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_F
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as lpips_F
from typing_extensions import override
from functools import partial
from numpy import ndarray


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
    
    per_epoch: List
    per_epoch_val: List

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self, 
        data_range: Optional[Tuple[float, float]] = None, 
        **kwargs: Any
        ) -> None:
        
        super().__init__(**kwargs)

        if isinstance(data_range, tuple):
            self._data_range = data_range
            self.data_range = data_range
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
        else:
            self._data_range = None
            self.data_range = None
            self.clamping_fn = None

        self.add_state("sum_metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self._reset_all()

    def _reset_all(self) -> None:
        self.per_epoch = []
        self.per_epoch_val = []

        if not self._data_range:
            self.data_range = None
            self.clamping_fn = None

        self.reset()

    def _pre_calc(self, 
                  preds: Tensor, 
                  target: Tensor
                  ) -> Tuple[Tensor, Tensor]:
        """ Some setup steps required prior to calculation. """

        # set data_range and clamping function if they aren't defined
        if self.data_range is None:
            self.data_range = (target.min().item(), target.max().item())
            self.clamping_fn = partial(torch.clamp, min=self.data_range[0], max=self.data_range[1])
        
        # clamp preds and target to data_range
        preds = self.clamping_fn(preds)
        target = self.clamping_fn(target)
        
        return preds, target
    
    def calculate(self, 
                  *_: Any) -> Tensor:
        raise NotImplementedError
    
    @override
    def update(self, 
               preds: Tensor, 
               target: Tensor, 
               ) -> None:
        """ Update state with preds and target. """

        self.sum_metric += self.calculate(preds, target)
        self.total += 1
    
    @override
    def forward(self, 
                preds: Tensor, 
                target: Tensor, 
                ) -> Tensor:
        """ Calculate metric and update state. """

        self.update(preds, target)
        return self.calculate(preds, target)
    
    def update_on_epoch(self, 
                        val: bool = False
                        ) -> None:
        """ Update epoch-level state variables. """

        if self.total > 0:
            if val:
                self.per_epoch_val.append((self.sum_metric / self.total).item())
            else:
                self.per_epoch.append((self.sum_metric / self.total).item())

    @override
    def compute(self, 
                ) -> Tensor:
        """ Compute metric over state and update epoch tracker. """

        return self.sum_metric / self.total

class MSE_Loss(CustomFittingMetric):
    """Averaging meter for tracking MSE-SSIM+1 loss over training and validation epochs. 
    Based on torchmetrics.regression.mse.MeanSquaredError class."""

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        data_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any
        ) -> None:
        
        super().__init__(data_range, **kwargs)
    
    @property
    def name(self) -> str:
        return "mse"

    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor
                  ) -> Tensor:
        """Calculate metric from preds and target. """

        preds, target = self._pre_calc(preds, target)

        return mse_F(preds, target)


class MSE_SSIM_Loss(CustomFittingMetric):
    """Averaging meter for tracking MSE-SSIM+1 loss over training and validation epochs. 
    Based on torchmetrics.image.ssim.StructuralSimilarityIndexMeasure and torchmetrics.regression.mse.MeanSquaredError classes."""

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        data_range: Optional[Tuple[float, float]] = None,
        k1: float = 0.01,
        k2: float = 0.03, 
        **kwargs
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
                  target: Tensor
                  ) -> Tensor:
        """ Calculate metric from preds and target. """
        
        preds, target = self._pre_calc(preds, target)
        
        mse = mse_F(preds, target)
        similarity = ssim_F(preds, target)
        
        return mse - similarity + 1


class MSE_LPIPS_Loss(CustomFittingMetric):
    """Averaging meter for tracking MSE+LPIPS loss over training and validation epochs. 
    Based on torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity and torchmetrics.regression.mse.MeanSquaredError classes."""

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

        self.net_type = net_type
        self.normalize = normalize
        if normalize:
            self.data_range = (0, 1)
        else:
            self.data_range = (-1, 1)
        self.clamping_fn = partial(torch.clamp, min=self.data_range[0], max=self.data_range[1])

    @property
    def name(self) -> str:
        return "mse-lpips"
        
    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor
                  ) -> Tensor:
                
        preds, target = self._pre_calc(preds, target)

        mse = mse_F(preds, target)
        lpips = lpips_F(preds.tile((1, 3, 1, 1)), 
                        target.tile((1, 3, 1, 1)), 
                        net_type=self.net_type, normalize=self.normalize)
        # combo = mse + lpips
        # combo.grad_fn = lpips.grad_fn

        return mse + lpips


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

        self.net_type = net_type
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
                  target: Tensor) -> Tensor:
        
        preds, target = self._pre_calc(preds, target)

        return lpips_F(preds.tile((1, 3, 1, 1)), 
                       target.tile((1, 3, 1, 1)), 
                       net_type=self.net_type, 
                       normalize=self.normalize)


class PSNR_Meter(CustomFittingMetric):
    """Averaging meter for tracking PSNR over training and validation epochs. 
    Based on torchmetrics.image.psnr.PeakSignalNoiseRatio class."""

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        data_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
        ) -> None:
        
        super().__init__(data_range, **kwargs)
    
    @property
    def name(self) -> str:
        return "psnr"

    @override
    def calculate(self, 
                  preds: Tensor, 
                  target: Tensor
                  ) -> Union[Tuple, Tensor]:
        """Calculate metric from preds and target. """

        preds, target = self._pre_calc(preds, target)

        # set requires_grad = False because this isn't a loss function and we can save memory
        return psnr_F(preds, target, data_range=self.data_range).requires_grad_(False)
    

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
        data_range: Optional[Tuple[float, float]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        **kwargs: Any
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
                  target: Tensor
                  ) -> Tensor:
        
        preds, target = self._pre_calc(preds, target)

        # set requires_grad = False because this isn't a loss function and we can save memory
        return ssim_F(preds, target, 
                      gaussian_kernel=self.gaussian_kernel, 
                      sigma=self.sigma, 
                      kernel_size=self.kernel_size, 
                      data_range=self.data_range, 
                      k1=self.k1, 
                      k2=self.k2).requires_grad_(False)

import numpy as np
from typing import Optional, Tuple
import torch
from torch.nn import functional as F


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: Optional[str] = None,
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if tails is None:
        outputs, logabsdet = rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
    else:
        outputs, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            tails=tails,
            tail_bound=tail_bound,
        )

    return outputs, logabsdet


def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bin_locations[..., bin_locations.size(-1) - 1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str = "linear",
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    mask_size = inside_interval_mask.size()  # (b, 1, d)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        # unnormalized_derivatives[..., -1] = constant
        unnormalized_derivatives[..., unnormalized_derivatives.size(-1) - 1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        # NOTE: operator 'masked_select' not implemented by ONNX
        # outputs[outside_interval_mask] = inputs.masked_select(outside_interval_mask)
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    n_elements = mask_size.numel()
    n_widths = unnormalized_widths.shape[-1]
    n_heights = unnormalized_heights.shape[-1]
    n_derivatives = unnormalized_derivatives.shape[-1]
    masked_inputs = inputs[inside_interval_mask].view(n_elements)
    masked_unnormalized_widths = (
        unnormalized_widths[inside_interval_mask, :]
        .view(n_elements, n_widths)
    )
    masked_unnormalized_heights = (
        unnormalized_heights[inside_interval_mask, :]
        .view(n_elements, n_heights)
    )
    masked_unnormalized_derivatives = (
        unnormalized_derivatives[inside_interval_mask, :]
        .view(n_elements, n_derivatives)
    )
    # NOTE: operator 'masked_select' not implemented by ONNX
    # masked_inputs = inputs.masked_select(inside_interval_mask)
    # m_inside_interval_mask = inside_interval_mask.view(*mask_size, 1)
    # masked_unnormalized_widths = (
    #     unnormalized_widths.masked_select(m_inside_interval_mask)
    #     .view(n_elements, n_widths)
    # )
    # masked_unnormalized_heights = (
    #     unnormalized_heights.masked_select(m_inside_interval_mask)
    #     .view(n_elements, n_heights)
    # )
    # masked_unnormalized_derivatives = (
    #     unnormalized_derivatives.masked_select(m_inside_interval_mask)
    #     .view(n_elements, n_derivatives)
    # )

    masked_outputs, masked_logabsdet = rational_quadratic_spline(
        inputs=masked_inputs,
        unnormalized_widths=masked_unnormalized_widths,
        unnormalized_heights=masked_unnormalized_heights,
        unnormalized_derivatives=masked_unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs[inside_interval_mask] = masked_outputs
    logabsdet[inside_interval_mask] = masked_logabsdet

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if torch.min(inputs) < left or torch.max(inputs) > right:
    #     raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    # if min_bin_width * num_bins > 1.0:
    #     raise ValueError("Minimal bin width too large for the number of bins")
    # if min_bin_height * num_bins > 1.0:
    #     raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    # cumwidths[..., -1] = right
    cumwidths[..., cumwidths.size(-1) - 1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    # cumheights[..., -1] = top
    cumheights[..., cumheights.size(-1) - 1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        # FIXME: changed to allow onnx export
        # assert (discriminant >= 0).all(), discriminant
        discriminant = discriminant.maximum(torch.Tensor([0.0]))

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet

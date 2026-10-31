import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


def _to_float_timestep(timestep) -> float:
    if torch.is_tensor(timestep):
        return float(timestep.flatten()[0].item())
    return float(timestep)


# ComfyUI model_sampling mixins that mark a rectified-flow / flow-matching model.
# ComfyUI builds model_sampling as `class ModelSampling(<sampling base>, <prediction type>)`
# (comfy/model_base.py:model_sampling) and itself tests for flow models with
# `isinstance(model_sampling, comfy.model_sampling.CONST)`:
#   CONST         - ModelType.FLOW / FLUX / FLOW_AV and the IMG_TO_IMG_FLOW / IMG_TO_IMG_VELOCITY
#                   subclasses (SD3, Flux 1/2, Wan, HunyuanVideo, Qwen-Image, Z-Image, LTX-Video, ...)
#   COSMOS_RFLOW  - ModelType.FLOW_COSMOS (Cosmos Predict2: rectified flow on an EDM-style sigma range)
_FLOW_SAMPLING_TYPES = ("CONST", "COSMOS_RFLOW")


def _model_sampling_is_flow(model_sampling) -> Optional[bool]:
    """Classify a ComfyUI model_sampling object; None if it can't be inspected.

    Matches on class names in the MRO rather than importing comfy.model_sampling so the
    check also works for model_sampling classes built by other custom nodes.
    """
    if model_sampling is None:
        return None
    try:
        names = {k.__name__ for k in type(model_sampling).__mro__}
    except Exception:
        return None
    return any(n in names for n in _FLOW_SAMPLING_TYPES)


def _model_sampling_sigma_range(model_sampling) -> Optional[Tuple[float, float]]:
    try:
        return float(model_sampling.sigma_min), float(model_sampling.sigma_max)
    except Exception:
        return None


def _sample_sigmas_from_c(c: dict) -> Optional[torch.Tensor]:
    """The full sampling schedule ComfyUI stores in transformer_options["sample_sigmas"].

    Note: transformer_options["sigmas"] is NOT the schedule - ComfyUI sets it to the current
    timestep (one value per batch item) in samplers._calc_cond_batch, so it isn't used here.
    """
    try:
        t_opts = c.get("transformer_options", None) if isinstance(c, dict) else None
        if not isinstance(t_opts, dict):
            return None
        sigmas = t_opts.get("sample_sigmas", None)
        if torch.is_tensor(sigmas) and sigmas.numel() > 1:
            return sigmas
    except Exception:
        pass
    return None


def _is_flow_schedule(c: dict) -> bool:
    """Last-resort heuristic used only when the model_sampling object is unavailable.

    Flow models (CONST) sample sigmas in [0, 1]. This can misfire on low-denoise img2img with
    eps models (every sigma < 1), which is why the model_sampling check takes priority.
    """
    sigmas = _sample_sigmas_from_c(c)
    if sigmas is None:
        return False
    try:
        return float(sigmas.detach().flatten().float().max().item()) <= 1.05
    except Exception:
        return False


def _schedule_minmax_from_c(c: dict) -> Optional[Tuple[float, float]]:
    sigmas = _sample_sigmas_from_c(c)
    if sigmas is None:
        return None
    try:
        s = sigmas.detach().flatten().float()
        return float(s.min().item()), float(s.max().item())
    except Exception:
        return None


def _normalize_to_unit_interval(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    denom = (t_max - t_min)
    if abs(denom) < 1e-12:
        return torch.zeros_like(t)
    x = (t - t_min) / denom
    return x


def _normalize_to_chebyshev_domain(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    x01 = _normalize_to_unit_interval(t, t_min=t_min, t_max=t_max)
    x = x01 * 2.0 - 1.0
    return x.clamp(-1.0, 1.0)


def _chebyshev_design_matrix(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Return Phi where Phi[i, k] = T_k(x_i), k in [0, degree-1]."""
    if degree <= 0:
        raise ValueError("degree must be >= 1")
    x = x.reshape(-1)
    n = x.shape[0]

    phi = x.new_empty((n, degree))
    phi[:, 0] = 1.0
    if degree == 1:
        return phi

    phi[:, 1] = x
    for k in range(2, degree):
        phi[:, k] = 2.0 * x * phi[:, k - 1] - phi[:, k - 2]
    return phi


def _fit_ridge(Phi: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """Solve ridge regression: argmin_C ||Phi C - Y||^2 + lam ||C||^2.

    Phi: [N, M]
    Y:   [N, D]
    Returns C: [M, D]
    """
    n, m = Phi.shape
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError("Y must be [N, D] with same N as Phi")

    lam_f = float(lam)
    if lam_f < 0:
        lam_f = 0.0

    # Use float32 math for stability even if model runs in fp16/bf16.
    Phi32 = Phi.float()
    Y32 = Y.float()

    # (Phi^T Phi + lam I) C = Phi^T Y
    A = Phi32.transpose(0, 1) @ Phi32
    if lam_f > 0:
        A = A + (lam_f * torch.eye(m, device=A.device, dtype=A.dtype))
    else:
        # Still add a tiny diagonal for numerical robustness.
        A = A + (1e-8 * torch.eye(m, device=A.device, dtype=A.dtype))
    B = Phi32.transpose(0, 1) @ Y32
    C = torch.linalg.solve(A, B)
    return C


def _forecast_from_coeffs(C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """C: [M, D], x: scalar tensor -> y_hat: [D]"""
    m = C.shape[0]
    phi = _chebyshev_design_matrix(x.reshape(1), degree=m).float()  # [1, M]
    y = (phi @ C).reshape(-1)  # [D]
    return y


@dataclass
class _SpectrumState:
    cache_ts: List[float] = field(default_factory=list)
    cache_y: List[torch.Tensor] = field(default_factory=list)
    coeffs: Optional[torch.Tensor] = None  # [M, D] float32
    step_index: int = 0
    last_t: Optional[float] = None
    last_shape: Optional[Tuple[int, ...]] = None
    last_dtype: Optional[torch.dtype] = None
    last_device: Optional[torch.device] = None

    def reset(self):
        self.cache_ts.clear()
        self.cache_y.clear()
        self.coeffs = None
        self.step_index = 0
        self.last_t = None
        self.last_shape = None
        self.last_dtype = None
        self.last_device = None


class MaxSpectrumPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "m": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "lam": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.001}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
                "model_type": (["auto", "eps", "flow"], {"default": "auto",
                    "tooltip": "Model parameterization. 'flow' covers rectified-flow / flow-matching "
                               "models (Flux 1/2, SD3, Wan, HunyuanVideo, Qwen-Image, Z-Image, "
                               "LTX-Video, Chroma, HiDream, Lumina 2, Cosmos Predict2, ...). 'eps' "
                               "covers eps/v-prediction/EDM models (SD 1.5, SDXL, SVD). 'auto' "
                               "reads the model's sampling type, the same way ComfyUI does."}),
            },
            "optional": {
                "real_final_step": ("BOOLEAN", {"default": True,
                    "tooltip": "Always run the real model on the last step of the schedule. That "
                               "step's output becomes the final latent, so forecasting it costs the "
                               "most quality - especially on short flow schedules (4-8 steps)."}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Always run the real model for the first N steps, where the output "
                               "changes fastest. 0 keeps the minimal warm-up (the first two real "
                               "passes the fit needs). Each extra step costs one model pass, so keep "
                               "it small on short schedules."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(self, model, w: float, m: int, lam: float, window_size: int, model_type: str = "auto",
              real_final_step: bool = True, warmup_steps: int = 0):
        model_clone = model.clone()

        # Keep separate Spectrum state per (cond_or_uncond) signature.
        states: Dict[Tuple[int, ...], _SpectrumState] = {}

        w_f = float(max(0.0, min(1.0, w)))
        degree = int(max(1, m))
        lam_f = float(max(0.0, lam))
        win = int(max(1, window_size))
        forced_flow = model_type == "flow"
        forced_eps = model_type == "eps"
        warmup = int(max(0, warmup_steps))

        # Model sampling as configured on this MODEL. A sampling node applied after this patch
        # (ModelSamplingSD3/Flux/AuraFlow, ...) replaces it, so the wrapper re-reads the live
        # object from the model at call time and only falls back to this one.
        try:
            patch_time_sampling = model.get_model_object("model_sampling")
        except Exception:
            patch_time_sampling = None
        debug: Dict[str, object] = {}

        def _get_state(key: Tuple[int, ...]) -> _SpectrumState:
            st = states.get(key)
            if st is None:
                st = _SpectrumState()
                states[key] = st
            return st

        def unet_wrapper_function(model_function, kwargs):
            # kwargs usually contains: input, timestep, c, cond_or_uncond
            x_in = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs.get("c", {})

            key_list = kwargs.get("cond_or_uncond", [])
            try:
                key = tuple(int(v) for v in key_list)
            except Exception:
                key = ()

            st = _get_state(key)

            t_val = _to_float_timestep(timestep)
            out_dtype = x_in.dtype if torch.is_tensor(x_in) else None
            out_device = x_in.device if torch.is_tensor(x_in) else None

            # Reset if a new sampling run likely started or if tensor layout changed.
            eps = 1e-7
            if st.last_t is not None and (t_val > st.last_t + eps):
                st.reset()
            if torch.is_tensor(x_in):
                shape = tuple(int(s) for s in x_in.shape)
                if st.last_shape is not None and shape != st.last_shape:
                    st.reset()
                if st.last_dtype is not None and out_dtype != st.last_dtype:
                    st.reset()
                if st.last_device is not None and out_device != st.last_device:
                    st.reset()
                st.last_shape = shape
                st.last_dtype = out_dtype
                st.last_device = out_device

            # Step index handling: prefer schedule-based index if available.
            schedule = _sample_sigmas_from_c(c)

            # model_function is BaseModel.apply_model, so its owner holds the live model_sampling.
            model_sampling = getattr(getattr(model_function, "__self__", None), "model_sampling", None)
            if model_sampling is None:
                model_sampling = patch_time_sampling

            # Detect model type for correct sigma-domain handling.
            if forced_flow:
                is_flow = True
            elif forced_eps:
                is_flow = False
            else:
                is_flow = _model_sampling_is_flow(model_sampling)
                if is_flow is None:
                    is_flow = _is_flow_schedule(c)
            debug["is_flow"] = is_flow

            is_final_step = False
            if schedule is not None:
                sigmas = schedule.detach().flatten()
                t0 = float(sigmas[0].item())
                # If we jumped back to the first sigma (or higher), treat as reset.
                if st.last_t is not None and abs(t_val - t0) < 1e-8 and (t_val > st.last_t + eps):
                    st.reset()

                # Find closest index; also handle non-exact matches.
                try:
                    target = torch.tensor([t_val], device=sigmas.device, dtype=sigmas.dtype)
                    diff = (sigmas - target).abs()
                    idx = int(diff.argmin().item())
                except Exception:
                    idx = st.step_index
                st.step_index = idx
                # The sampler evaluates sigmas[0 .. n-2] and steps to sigmas[n-1] (usually 0).
                is_final_step = idx >= sigmas.numel() - 2
            else:
                if st.last_t is None:
                    st.step_index = 0
                else:
                    if abs(t_val - st.last_t) > 1e-8:
                        st.step_index += 1

            st.last_t = t_val

            def _resolve_tminmax(extra_t=None):
                """Return (t_min, t_max) for Chebyshev domain normalization.

                Priority: sampling schedule from context -> the model's own sigma range ->
                flow-model canonical [0, 1] -> cache-derived range.
                """
                tminmax = _schedule_minmax_from_c(c)
                if tminmax is not None:
                    return tminmax
                ms_range = _model_sampling_sigma_range(model_sampling)
                if ms_range is not None:
                    # Flow schedules run down to sigma = 0 (ComfyUI appends it), not sigma_min.
                    return (0.0 if is_flow else ms_range[0]), ms_range[1]
                if is_flow:
                    return 0.0, 1.0
                pts = list(st.cache_ts)
                if extra_t is not None:
                    pts = pts + [extra_t]
                if not pts:
                    return 0.0, 1.0
                return float(min(pts)), float(max(pts))

            # Decide whether to do a real model pass.
            must_run_real = (
                (st.step_index < warmup)
                or (win <= 1)
                or (st.step_index % win == 0)
                or (len(st.cache_ts) < 2)
                or (real_final_step and is_final_step)
            )
            if must_run_real:
                y = model_function(x_in, timestep, **c)
                if not torch.is_tensor(y):
                    return y

                # Update cache with detached tensor to avoid holding onto any graph.
                st.cache_ts.append(t_val)
                st.cache_y.append(y.detach())

                # Keep only a small history; enough points to fit degree.
                max_cache = max(2, degree + 1)
                if len(st.cache_ts) > max_cache:
                    st.cache_ts = st.cache_ts[-max_cache:]
                    st.cache_y = st.cache_y[-max_cache:]

                # Fit coefficients once we have at least 2 points.
                try:
                    t_min, t_max = _resolve_tminmax()
                    ts = torch.tensor(st.cache_ts, device=y.device, dtype=torch.float32)
                    xs = _normalize_to_chebyshev_domain(ts, t_min=t_min, t_max=t_max)
                    Phi = _chebyshev_design_matrix(xs, degree=degree)

                    # Fit the history *relative to the newest real output*. The ridge penalty
                    # then shrinks the forecast toward that output (plain caching) instead of
                    # toward zero, which biased every forecast - noticeably on flow models.
                    anchor = st.cache_y[-1].float().reshape(-1)
                    Y = torch.stack([v.float().reshape(-1) - anchor for v in st.cache_y], dim=0)  # [N, D]
                    st.coeffs = _fit_ridge(Phi, Y, lam=lam_f)
                except Exception:
                    st.coeffs = None

                return y

            # Forecast pass.
            if st.coeffs is None or len(st.cache_y) == 0:
                return model_function(x_in, timestep, **c)

            try:
                last_real = st.cache_y[-1]

                t_min, t_max = _resolve_tminmax(extra_t=t_val)
                x_t = _normalize_to_chebyshev_domain(
                    torch.tensor([t_val], device=last_real.device, dtype=torch.float32),
                    t_min=t_min,
                    t_max=t_max,
                )
                pred_flat = last_real.float().reshape(-1) + _forecast_from_coeffs(st.coeffs, x_t)
                pred = pred_flat.reshape(last_real.shape).to(dtype=last_real.dtype)

                if not torch.isfinite(pred).all():
                    return model_function(x_in, timestep, **c)

                if w_f >= 1.0:
                    return pred
                if w_f <= 0.0:
                    return last_real
                return (w_f * pred) + ((1.0 - w_f) * last_real)
            except Exception:
                return model_function(x_in, timestep, **c)

        unet_wrapper_function.spectrum_debug = debug
        model_clone.set_model_unet_function_wrapper(unet_wrapper_function)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "MaxSpectrumPatcher": MaxSpectrumPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaxSpectrumPatcher": "Spectrum Patcher (Max)",
}

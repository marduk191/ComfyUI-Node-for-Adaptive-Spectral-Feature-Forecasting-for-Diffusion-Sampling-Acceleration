"""End-to-end test of the Spectrum patcher against a real ComfyUI checkout.

Every model registered in ``comfy.supported_models`` is instantiated with its
real ``model_base`` class, ``model_sampling`` object and latent format, but the
diffusion network is swapped for a tiny deterministic stand-in, so no weights
are needed. Each model is then sampled through ComfyUI's own KSampler with and
without the patch.

Usage:
    python tests/test_flow_models.py --comfyui /path/to/ComfyUI [--quick] [-v]

Exit code is non-zero if any check fails.
"""

import argparse
import importlib.util
import inspect
import logging
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Models registered in supported_models that are not sampled with KSampler
# (detection / segmentation / depth backbones that reuse the FLOW enum).
NON_GENERATIVE = {"RT_DETR_v4", "SAM3", "SAM31", "DepthAnything3"}


SAMPLERS = ["euler", "dpmpp_2m", "heun", "res_multistep", "uni_pc_bh2"]
SCHEDULERS = ["simple", "beta", "normal", "sgm_uniform"]
# Known limitations: reported as XFAIL instead of failing the run.
KNOWN_LIMITATIONS = {
    # EDM v-prediction with sigma up to 700: the polynomial is fitted on raw sigma, so most steps
    # sit at one end of the Chebyshev domain. Not a flow model.
    # Stable Audio 1.0 (V_PREDICTION_CONTINUOUS, sigma up to 500) behaves the same. On these
    # ranges even plain last-output caching drifts far at 8 steps; the forecast is about on par.
    "SVD_img2vid": "EDM sigma range (0.002..700): skipping steps is lossy even with plain caching",
    "SV3D_u": "EDM sigma range (0.002..700): skipping steps is lossy even with plain caching",
    "SV3D_p": "EDM sigma range (0.002..700): skipping steps is lossy even with plain caching",
    "StableAudio": "continuous-v sigma range (0.03..500): skipping steps is lossy even with plain caching",
    # Rectified flow on an EDM-style sigma range (0.002..80). Forecasting is roughly on par with
    # plain output caching here; fitting on flow time sigma/(sigma+1) instead tested worse.
    "CosmosT2IPredict2": "forecast ~= last-output caching on the 0.002..80 sigma range",
}

# A representative flow model per sampling backend, used for the full sampler/scheduler matrix.
MATRIX_MODELS = ["ZImage", "Flux2", "WAN22_T2V", "CosmosT2IPredict2", "MiniMaxH3", "WAN21_FlowRVS"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", default=os.environ.get("COMFYUI_PATH"), help="path to a ComfyUI checkout")
    p.add_argument("--node", default=REPO, help="path to this custom node (folder containing nodes.py)")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="skip the sampler/scheduler matrix")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if not args.comfyui:
        p.error("--comfyui (or COMFYUI_PATH) is required")
    return args


ARGS = parse_args()
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, os.path.abspath(ARGS.comfyui))
logging.basicConfig(level=logging.WARNING)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

import comfy.model_base  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.supported_models  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)

# ComfyUI has its own top-level nodes.py, so load ours under a unique name.
_spec = importlib.util.spec_from_file_location("max_spectrum_nodes", os.path.join(ARGS.node, "nodes.py"))
spectrum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spectrum)


class TinyNet(torch.nn.Module):
    """Deterministic, smooth stand-in for a diffusion transformer."""

    calls = 0
    out_ch = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x, timesteps=None, *args, **kwargs):
        TinyNet.calls += 1
        if isinstance(x, (list, tuple)):
            return [self._f(v, timesteps) for v in x]
        return self._f(x, timesteps)

    def _f(self, x, timesteps):
        t = timesteps
        if not torch.is_tensor(t):
            t = torch.tensor([0.5 if t is None else float(t)])
        t = t.float().reshape(-1)[:1].reshape([1] * x.ndim).to(x.device)
        tn = t / (1.0 + t.abs())
        if TinyNet.out_ch is not None and x.ndim > 1 and x.shape[1] > TinyNet.out_ch:
            x = x[:, :TinyNet.out_ch]  # concat-conditioned models: predict latent channels only
        return (torch.tanh(x) * (0.6 + 0.3 * tn) + 0.1 * torch.sin(3.0 * tn)) * self.scale.to(x.dtype)


_orig_init = comfy.model_base.BaseModel.__init__


def _tiny_init(self, model_config, *a, **kw):
    model_config.unet_config["disable_unet_model_creation"] = True
    _orig_init(self, model_config, *a, **kw)
    self.diffusion_model = TinyNet()


comfy.model_base.BaseModel.__init__ = _tiny_init

CPU = torch.device("cpu")


def _make_cfg(cfg_name):
    cls = getattr(comfy.supported_models, cfg_name)
    unet_config = dict(cls.unet_config)
    unet_config.setdefault("in_channels", 4)
    unet_config.setdefault("hidden_size", 3072)  # read by Flux2/Flux3 configs for memory estimates
    return cls(unet_config)


def _patcher(model):
    return comfy.model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


_BUILD_CACHE = {}


def _declared_model_type(cfg_name):
    """ModelType default of the model_base class a config's get_model() instantiates."""
    for klass in inspect.getmro(getattr(comfy.supported_models, cfg_name)):
        if "get_model" not in klass.__dict__:
            continue
        for base_name in re.findall(r"model_base\.(\w+)\(", inspect.getsource(klass.get_model)):
            for k in inspect.getmro(getattr(comfy.model_base, base_name)):
                if "__init__" in k.__dict__:
                    p = inspect.signature(k.__init__).parameters.get("model_type")
                    if p is not None and p.default is not inspect.Parameter.empty:
                        return p.default
    return comfy.model_base.ModelType.EPS


def build(cfg_name):
    """Return (model_config, ModelPatcher, how) for a registered ComfyUI model.

    how == "real":    the model's own model_base class (extra_conds, _apply_model overrides).
    how == "generic": a plain BaseModel carrying the model's real model_sampling object and
                      latent format; used when the real class needs model-specific inputs
                      (reference audio, concat images, ...) that this harness doesn't fake.
    """
    if cfg_name in _BUILD_CACHE:
        return _BUILD_CACHE[cfg_name]
    cfg = _make_cfg(cfg_name)
    try:
        real = cfg.get_model({}, "", device=CPU)
        model_type = real.model_type
    except Exception:
        # The real class needs weights to construct (e.g. StableAudio3's seconds embedder).
        real, model_type = None, _declared_model_type(cfg_name)
    try:
        if real is None:
            raise RuntimeError("no real model")
        run(_patcher(real), cfg, steps=2)
        out = (cfg, _patcher(real), "real")
    except Exception:
        cfg = _make_cfg(cfg_name)
        generic = comfy.model_base.BaseModel(cfg, model_type=model_type, device=CPU)
        if real is not None:
            generic.model_sampling = real.model_sampling
        out = (cfg, _patcher(generic), "generic")
    _BUILD_CACHE[cfg_name] = out
    return out


def mro_names(obj):
    return {k.__name__ for k in type(obj).__mro__}


def is_comfy_flow(model_sampling):
    """ComfyUI's own definition (see comfy/k_diffusion/sampling.py, nodes_model_advanced.py)."""
    return bool(mro_names(model_sampling) & {"CONST", "COSMOS_RFLOW"})


def make_latent(cfg):
    lf = cfg.latent_format
    ch = getattr(lf, "latent_channels", 4)
    dims = getattr(lf, "latent_dimensions", 2)
    spatial = {1: [16], 2: [8, 8], 3: [2, 8, 8]}.get(dims, [8, 8])
    return torch.zeros([1, ch] + spatial)


def make_cond(seed):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(1, 4, 64, generator=g), {"pooled_output": torch.randn(1, 64, generator=g), "guidance": 3.5}]]


def run(patcher, cfg, *, steps, cfg_scale=1.0, sampler="euler", scheduler="simple", denoise=1.0, latent=None):
    latent = make_latent(cfg) if latent is None else latent
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(0))
    TinyNet.calls = 0
    TinyNet.out_ch = latent.shape[1]
    out = comfy.sample.sample(
        patcher, noise, steps, cfg_scale, sampler, scheduler,
        make_cond(1), make_cond(2), latent, denoise=denoise, disable_pbar=True, seed=0,
    )
    return out, TinyNet.calls


def patch(patcher, **kw):
    params = dict(w=0.5, m=4, lam=0.1, window_size=2, model_type="auto")
    params.update(kw)
    return spectrum.MaxSpectrumPatcher().patch(patcher, **params)[0]


def detected_flow(patched):
    fn = patched.model_options.get("model_function_wrapper")
    dbg = getattr(fn, "spectrum_debug", None)
    return None if dbg is None else dbg.get("is_flow")


def rel_err(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / (b.norm() + 1e-8))


class Results:
    def __init__(self):
        self.rows = []
        self.failures = []
        self.xfails = []

    def check(self, name, ok, detail="", known=None):
        self.rows.append((name, ok, detail))
        status = "PASS" if ok else ("XFAIL" if known else "FAIL")
        if status == "XFAIL":
            self.xfails.append((name, known))
        elif not ok:
            self.failures.append((name, detail))
        if ARGS.verbose or status != "PASS":
            print(f"  [{status}] {name} {detail}")


def test_model(res, name, steps, expect_flow):
    try:
        cfg, base, how = build(name)
    except Exception as e:
        res.check(f"{name}: build", False, repr(e))
        return
    ms = base.get_model_object("model_sampling")
    kind = "/".join(sorted(mro_names(ms) & {"CONST", "COSMOS_RFLOW", "EPS", "V_PREDICTION", "EDM", "X0", "V_PREDICTION_DDPM", "IMG_TO_IMG"}))
    res.check(f"{name} [{how}, {type(ms).__mro__[1].__name__}/{kind}]: ComfyUI classifies as "
              f"{'flow' if expect_flow else 'non-flow'}", is_comfy_flow(ms) == expect_flow)

    for cfg_scale in (1.0, 4.0):
        try:
            ref, ref_calls = run(base, cfg, steps=steps, cfg_scale=cfg_scale)
            p = patch(base)
            out, calls = run(p, cfg, steps=steps, cfg_scale=cfg_scale)
        except Exception as e:
            res.check(f"{name}: sample cfg={cfg_scale}", False, f"{e!r}\n{traceback.format_exc() if ARGS.verbose else ''}")
            continue
        err = rel_err(out, ref)
        tag = f"{name} cfg={cfg_scale}"
        res.check(f"{tag}: finite + same shape", bool(torch.isfinite(out).all()) and out.shape == ref.shape)
        res.check(f"{tag}: fewer model calls ({calls} < {ref_calls})", calls < ref_calls)
        res.check(f"{tag}: close to unpatched (rel err {err:.4f})", err < 0.25, known=KNOWN_LIMITATIONS.get(name))
        det = detected_flow(p)
        if det is not None:
            res.check(f"{tag}: auto-detected flow={det}", det == expect_flow)


def test_matrix(res, steps):
    """Sampler x scheduler matrix on one model per flow sampling backend.

    Absolute error against the unpatched run depends on the stand-in network, so the check
    here is relative: with the same call budget, the Spectrum forecast (w=0.5) must not be
    worse than plain caching of the last real output (w=0).
    """
    for name in MATRIX_MODELS:
        cfg, base, _ = build(name)
        p = patch(base)
        naive = patch(base, w=0.0)
        for sampler in SAMPLERS:
            for sched in SCHEDULERS:
                tag = f"{name} {sampler}/{sched}"
                try:
                    ref, ref_calls = run(base, cfg, steps=steps, cfg_scale=3.0, sampler=sampler, scheduler=sched)
                    out, calls = run(p, cfg, steps=steps, cfg_scale=3.0, sampler=sampler, scheduler=sched)
                    cached, _ = run(naive, cfg, steps=steps, cfg_scale=3.0, sampler=sampler, scheduler=sched)
                except Exception as e:
                    res.check(tag, False, repr(e))
                    continue
                err, err_cache = rel_err(out, ref), rel_err(cached, ref)
                ok = bool(torch.isfinite(out).all()) and calls < ref_calls and err <= err_cache * 1.05 + 1e-4
                res.check(f"{tag}: calls {calls}/{ref_calls}, rel err {err:.4f} (last-output cache {err_cache:.4f})", ok,
                          known=KNOWN_LIMITATIONS.get(name))


def test_edge_cases(res, steps):
    # 1. Low-denoise img2img on an eps model: every sigma is < 1, which fooled the old
    #    sigma-range heuristic into flow mode.
    cfg, base, _ = build("SDXL")
    p = patch(base)
    latent = torch.randn(make_latent(cfg).shape, generator=torch.Generator().manual_seed(5))
    try:
        run(p, cfg, steps=steps, denoise=0.12, latent=latent)
        det = detected_flow(p)
        if det is not None:
            res.check(f"SDXL denoise=0.12: auto-detected flow={det}", det is False)
    except Exception as e:
        res.check("SDXL denoise=0.12", False, repr(e))

    # 2. A shift node (ModelSamplingSD3) applied *after* the patcher must be honoured.
    try:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        cfg, base, _ = build("SDXL")
        p = patch(base)
        p2 = ModelSamplingSD3().patch(p, shift=3.0)[0]
        run(p2, cfg, steps=steps)
        det = detected_flow(p2)
        if det is not None:
            res.check(f"SDXL + ModelSamplingSD3 after patch: auto-detected flow={det}", det is True)
    except Exception as e:
        res.check("SDXL + ModelSamplingSD3 after patch", False, repr(e))

    # 3. A batch of 2 latents: transformer_options["sigmas"] then has 2 entries (the current
    #    sigma per sample), which must not be mistaken for the sampling schedule.
    cfg, base, _ = build("ZImage")
    p = patch(base)
    lat = make_latent(cfg).repeat(2, 1, 1, 1)
    try:
        ref, ref_calls = run(base, cfg, steps=steps, latent=lat)
        out, calls = run(p, cfg, steps=steps, latent=lat)
        res.check(f"ZImage batch=2: calls {calls}/{ref_calls}, rel err {rel_err(out, ref):.4f}",
                  calls < ref_calls and rel_err(out, ref) < 0.25)
    except Exception as e:
        res.check("ZImage batch=2", False, repr(e))

    # 4. Forced modes still run.
    for mt in ("flow", "eps"):
        cfg, base, _ = build("Flux")
        try:
            out, _ = run(patch(base, model_type=mt), cfg, steps=steps)
            res.check(f"Flux model_type={mt}: finite", bool(torch.isfinite(out).all()))
        except Exception as e:
            res.check(f"Flux model_type={mt}", False, repr(e))

    # 5. Wan 2.2 style two-stage sampling (KSamplerAdvanced start/end steps, one patcher per
    #    stage). KSampler slices the schedule, so each stage's last step must run for real.
    cfg, base, _ = build("WAN22_T2V")
    for final in (True, False):
        try:
            total = 16
            latent = make_latent(cfg)
            noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(0))
            stage_calls = []
            for start, last in ((0, 8), (8, 16)):
                TinyNet.calls, TinyNet.out_ch = 0, latent.shape[1]
                latent = comfy.sample.sample(
                    patch(base, real_final_step=final), noise, total, 3.0, "euler", "simple",
                    make_cond(1), make_cond(2), latent, disable_noise=start > 0, start_step=start,
                    last_step=last, force_full_denoise=last == total, disable_pbar=True, seed=0,
                )
                stage_calls.append(TinyNet.calls)
            # 8 steps per stage, window 2: real at idx 0, 1 (warm-up), 2, 4, 6, plus 7 (final step).
            expected = [6, 6] if final else [5, 5]
            res.check(f"WAN22 two-stage real_final_step={final}: calls per stage {stage_calls} (expect {expected})",
                      stage_calls == expected and bool(torch.isfinite(latent).all()))
        except Exception as e:
            res.check(f"WAN22 two-stage real_final_step={final}", False, repr(e))

    # 6. Two consecutive runs on the same patched model (state must reset between runs).
    cfg, base, _ = build("Flux2")
    p = patch(base)
    try:
        a, _ = run(p, cfg, steps=steps)
        b, _ = run(p, cfg, steps=steps)
        res.check("Flux2 repeated run is deterministic", torch.allclose(a, b))
    except Exception as e:
        res.check("Flux2 repeated run", False, repr(e))


def main():
    t0 = time.time()
    res = Results()
    flow_models, eps_models = [], []
    for cls in comfy.supported_models.models:
        name = cls.__name__
        if name in NON_GENERATIVE:
            continue
        try:
            _, pt, _ = build(name)
        except Exception as e:
            res.check(f"{name}: build", False, repr(e))
            continue
        (flow_models if is_comfy_flow(pt.get_model_object("model_sampling")) else eps_models).append(name)

    print(f"ComfyUI flow models ({len(flow_models)}): {', '.join(flow_models)}")
    print(f"ComfyUI non-flow models ({len(eps_models)}): {', '.join(eps_models)}\n")
    # Every registered generative model: auto must pick flow for flow models and eps for the rest.
    for name in flow_models:
        test_model(res, name, ARGS.steps, expect_flow=True)
    for name in eps_models:
        test_model(res, name, ARGS.steps, expect_flow=False)
    test_edge_cases(res, ARGS.steps)
    if not ARGS.quick:
        test_matrix(res, ARGS.steps)

    n_ok = sum(1 for _, ok, _ in res.rows if ok)
    print(f"\n{n_ok}/{len(res.rows)} checks passed, {len(res.xfails)} known limitations (XFAIL), "
          f"{len(res.failures)} failed in {time.time() - t0:.1f}s")
    if res.failures:
        print(f"{len(res.failures)} failed:")
        for name, detail in res.failures:
            print(f"  - {name} {detail}".rstrip())
    return 1 if res.failures else 0


if __name__ == "__main__":
    sys.exit(main())

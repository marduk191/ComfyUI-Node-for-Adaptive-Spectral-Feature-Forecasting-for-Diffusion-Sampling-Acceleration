# Spectrum Patcher — ComfyUI Model Patch
<img width="1200" height="720" alt="spectrum-patcher-guide" src="https://github.com/user-attachments/assets/17b6c2f6-a7e7-418f-b117-dc548fbf5532" />

Training-free diffusion acceleration via Chebyshev spectral forecasting (in the style of *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*).

Original Spectrum implementation/research repo:
https://github.com/hanjq17/Spectrum

This custom node is a **MODEL patcher**: it wraps the model UNet/DiT forward via `set_model_unet_function_wrapper(...)` so it can work with normal ComfyUI samplers (Euler, DPM++, etc.).

## Install

Clone into `ComfyUI/custom_nodes`.

Recommended (clone into a correctly-named folder):

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/maximilianwicen/ComfyUI-Node-for-Adaptive-Spectral-Feature-Forecasting-for-Diffusion-Sampling-Acceleration.git ComfyUI-Max-Spectrum
```

Then restart ComfyUI.

## Node

- Menu name: **Spectrum Patcher (Max)**
- Category: `model/patches`
- Internal id: `MaxSpectrumPatcher`

## Compatible Models

Works with any model that uses ComfyUI's standard MODEL patcher interface:

- **Standard diffusion**: SD 1.5, SDXL, and similar eps/v-prediction/EDM models
- **Rectified-flow / flow-matching**: every flow model ComfyUI ships, listed below

### Flow models

`model_type = auto` decides "flow" the same way ComfyUI does (`isinstance(model_sampling, comfy.model_sampling.CONST)`, plus `COSMOS_RFLOW`). It reads the model's live `model_sampling` at sampling time, so a shift node such as `ModelSamplingSD3`, `ModelSamplingFlux` or `ModelSamplingAuraFlow` placed after this patcher is still picked up.

The table lists every flow model registered in ComfyUI's `comfy/supported_models.py` (audited against [Comfy-Org/ComfyUI@`0dbfd49`](https://github.com/Comfy-Org/ComfyUI/tree/0dbfd493b4cc604db743c699f0f7ce01348e54a4), 2026-10-31). Every row is covered by `tests/test_flow_models.py`.

| Family | ComfyUI model configs | Sampling |
|---|---|---|
| Flux.1 (dev / schnell / fill) | `Flux`, `FluxSchnell`, `FluxInpaint` | `ModelSamplingFlux` |
| Flux.2, Flux 3 | `Flux2`, `Flux3` | `ModelSamplingFlux` |
| Chroma, Chroma Radiance | `Chroma`, `ChromaRadiance` | `ModelSamplingFlux` |
| Qwen-Image | `QwenImage`, `QwenImage21` | `ModelSamplingFlux` |
| LongCat-Image, Krea 2, FIBO 2, Microsoft Lens | `LongCatImage`, `Krea2`, `Fibo2`, `Lens` | `ModelSamplingFlux` |
| LTX-Video, LTX audio+video | `LTXV`, `LTXAV` | `ModelSamplingFlux` |
| SD3 / SD3.5, AuraFlow | `SD3`, `AuraFlow` | `ModelSamplingDiscreteFlow` |
| Z-Image (Turbo), Z-Image pixel-space, Ming-Image, Lumina 2 | `ZImage`, `ZImagePixelSpace`, `MingImage`, `Lumina2` | `ModelSamplingDiscreteFlow` |
| HiDream, HiDream-O1 | `HiDream`, `HiDreamO1` | `ModelSamplingDiscreteFlow` |
| HunyuanImage 2.1 (+ refiner) | `HunyuanImage21`, `HunyuanImage21Refiner` | `ModelSamplingDiscreteFlow` |
| OmniGen2, Boogu | `Omnigen2`, `Boogu` | `ModelSamplingDiscreteFlow` |
| Kandinsky 5 (image / video) | `Kandinsky5Image`, `Kandinsky5` | `ModelSamplingDiscreteFlow` |
| Other image DiTs | `MageFlow`, `JoyImage`, `Ideogram4`, `MOSSImage`, `Anima`, `ErnieImage`, `PiD`, `PixelDiTT2I`, `SenseNovaU15` | `ModelSamplingDiscreteFlow` |
| Wan 2.1 | `WAN21_T2V`, `WAN21_I2V`, `WAN21_FunControl2V`, `WAN21_Vace`, `WAN21_Camera`, `WAN21_HuMo`, `WAN21_SCAIL`, `WAN21_SCAIL2`, `WAN21_CausalAR_T2V` | `ModelSamplingDiscreteFlow` |
| Wan 2.2 and later | `WAN22_T2V`, `WAN22_S2V`, `WAN22_Camera`, `WAN22_Animate`, `WAN22_WanDancer`, `WAN_Animate2` | `ModelSamplingDiscreteFlow` |
| Wan 2.1 FlowRVS | `WAN21_FlowRVS` | `ModelSamplingDiscrete` + `IMG_TO_IMG_FLOW` |
| HunyuanVideo (T2V / I2V / SkyReels), HunyuanVideo 1.5 (+ SR) | `HunyuanVideo`, `HunyuanVideoI2V`, `HunyuanVideoSkyreelsI2V`, `HunyuanVideo15`, `HunyuanVideo15_SR_Distilled` | `ModelSamplingDiscreteFlow` |
| Mochi, SeedVR2 | `GenmoMochi`, `SeedVR2` | `ModelSamplingDiscreteFlow` |
| MiniMax H3 (audio+video) | `MiniMaxH3` | `ModelSamplingAV` |
| Cosmos Predict2 (T2I / I2V) | `CosmosT2IPredict2`, `CosmosI2VPredict2` | `ModelSamplingCosmosRFlow` |
| Audio: Stable Audio 3, ACE-Step (1.0 / 1.5), MiniMax Music 3, YuE 2 | `StableAudio3`, `ACEStep`, `ACEStep15`, `MiniMaxMusic3`, `YuE2` | `ModelSamplingDiscreteFlow` |
| 3D: Hunyuan3D 2.0 / 2.0 mini / 2.1, TripoSplat, Trellis 2 | `Hunyuan3Dv2`, `Hunyuan3Dv2mini`, `Hunyuan3Dv2_1`, `TripoSplat`, `Trellis2` | `ModelSamplingDiscreteFlow` |

### Non-flow models

`auto` picks `eps` for all 30 non-flow models ComfyUI registers:

- SD 1.x/2.x and variants: `SD15`, `SD15_instructpix2pix`, `SD20`, `SD21UnclipL`, `SD21UnclipH`, `Stable_Zero123`, `LotusD`, `SD_X4Upscaler`
- SDXL and variants: `SDXL`, `SDXLRefiner`, `SDXL_instructpix2pix`, `SSD1B`, `KOALA_700M`, `KOALA_1B`, `Segmind_Vega`
- DiTs: `PixArtAlpha`, `PixArtSigma`, `HunyuanDiT`, `HunyuanDiT1`
- Stable Cascade: `Stable_Cascade_C`, `Stable_Cascade_B`
- Video and 3D: `CosmosT2V`, `CosmosI2V` (Cosmos 1.0, EDM), `CogVideoX_T2V`, `CogVideoX_I2V`, `CogVideoX_Inpaint`, `SVD_img2vid`, `SV3D_u`, `SV3D_p`
- Audio: `StableAudio` (1.0)

All of them run correctly with the patch. **SVD, SV3D and Stable Audio 1.0** are the exception on quality. Their sigmas reach 700 (500 for Stable Audio), and over such a wide range, skipping steps is lossy even with plain reuse of the last output; the forecast is roughly on par with that. Expect little benefit from this node on them, especially at low step counts.

Results from the test harness (a stand-in network, so treat these as relative numbers, not image quality):

- All 74 flow configs, both `cfg=1` and `cfg=4`: model calls drop to 75% at 8 steps with `window_size=2`, and the result stays within 2.5% of the unpatched run.
- The forecast is at least as accurate as plain "reuse the last output" caching in 115/120 sampler×scheduler combinations (euler, dpmpp_2m, heun, res_multistep, uni_pc_bh2 × simple, beta, normal, sgm_uniform).
- **Cosmos Predict2** is the exception. Its sigmas span 0.002–80, and the forecast there comes out about even with plain caching. It still runs correctly, but expect a smaller gain than on other flow models.

### Flow model tips

- Few-step / distilled flow models (Z-Image Turbo, Flux schnell, Wan or Qwen lightning LoRAs, 4–8 steps): `window_size` `2`, `m` `3–4`, `w` `0.5–0.8`, and keep `real_final_step` on.
- Longer flow schedules (20–50 steps on Flux, Wan, HunyuanVideo, Qwen-Image): `window_size` `3–4` saves more passes.
- Wan 2.2 two-expert (high/low noise) workflows: put one patcher on each model. Each expert's final step is run for real, along with its first two (warm-up). A stage of only ~4 steps therefore has little or nothing left to forecast. The patcher pays off from ~8 steps per stage, or use `window_size` `3`.
- Use any flow-compatible sampler and scheduler (euler, dpmpp_2m, res_multistep, uni_pc, ...). All of them are covered by the tests.

## Parameters

- `window_size`: run a real model pass every Nth step, forecast the others
- `m`: Chebyshev basis count / polynomial degree
- `lam`: ridge regularization strength. Higher values pull the forecast toward the last real output (plain caching). Lower values trust the polynomial more.
- `w`: blend factor between forecast and last real output on forecast steps
- `model_type`: `auto` (default) reads the model's sampling type, as described above. `flow` and `eps` force a mode.
- `real_final_step` (optional, default on): always run the real model on the last step of the schedule. That step's output becomes the final latent, so forecasting it costs the most quality, especially on short flow schedules.
- `warmup_steps` (optional, default 0): always run the real model for the first N steps of the schedule, where the output changes fastest. `0` keeps the minimal warm-up: the first two real passes the fit needs. Every extra warm-up step costs a model pass, so keep it at `0–2` on 4–8 step flow schedules. Longer runs have room for more warm-up.

## Testing

`tests/test_flow_models.py` runs this node through ComfyUI's real sampler, with no model weights needed. It builds every registered ComfyUI model with its own `model_base` class, `model_sampling` and latent format, but swaps the diffusion network for a tiny deterministic stand-in. A plain `BaseModel` carrying the model's real `model_sampling` is used when the real class needs inputs the harness doesn't fake. For each model it checks flow detection against ComfyUI, the call savings, and how far the result drifts from the unpatched run.

```bash
python tests/test_flow_models.py --comfyui /path/to/ComfyUI
```

Add `--quick` to skip the sampler×scheduler matrix, or `-v` to print every check.

## Notes

This implementation forecasts the output of ComfyUI's `BaseModel.apply_model`, which is the **denoised (x0) prediction** for every model type (for flow models: `x - sigma * velocity`). The forecast is based on a short history of real evaluations and is fitted relative to the most recent real output. It is conservative by design: if the fit fails or produces NaNs/Infs, it falls back to a real model call.

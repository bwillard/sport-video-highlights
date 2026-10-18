# Hardware for ML (tracking, ball, identity)

**Status:** Undecided (2026-10-18). CPU benchmark done (table below): players are workable overnight on the CPU, the ball at the resolution it needs is not (~18 h/game every frame).
**Why it matters:** Phases 3.2–4 run detection on every frame of whole games, need higher resolution for the ball, and involve a lot of trial and error (settings, re-runs, training a ball model). Turnaround time decides how fast that goes.

## The box today
- MSI B450M Bazooka Max WiFi (micro-ATX), AMD Ryzen 3 3200G (4 cores, no SMT), 123 GB RAM.
- Vega iGPU: VA-API video encode/decode only; can't run the models (no CUDA, no usable ROCm).
- With the 3200G, the x16 slot runs at PCIe 3.0 x8.

## Options
Speeds are estimates until measured; fill in the table at the bottom.

### 1) This hardware as it is
- YOLO11n @640: maybe 8–15 fps. A 75-min game (~135k frames) every frame: ~3–4 h.
- Ball needs @1280 and every frame: ~4× slower, most of a day per game. Training a ball model: days per attempt.
- Workable compromise: players on every 3rd frame (~1–1.5 h/game), stock ball detection on a centre crop only. Ball tracking noticeably weaker.
- Hidden cost: overnight runs mean about one experiment per day.
- Cheap step up: drop-in Ryzen 7 5700X (8C/16T, needs a BIOS update) for ~2.5–3× CPU speed. Still far behind a GPU, and it has no iGPU, so the box would run headless.

### 2) Add a GPU (e.g. RTX 5060 Ti 16GB)
- The 16 GB card is the **5060 Ti**; the plain 5060 is 8 GB only.
- YOLO11s @1280 at a few hundred fps on the card. The 3200G becomes the bottleneck (the tracker's camera-motion step runs on the CPU; decode and preprocessing must stay on the GPU). Realistic: ~20–60 min per game, every frame, full resolution.
- Ball model training: well under an hour. Re-ID embeddings and jersey OCR become fast too.
- Fit checklist: PSU wattage and an 8-pin PCIe connector (check the specific card; Nvidia suggests ~600 W total), case room for a 2-slot card ~20–30 cm long. PCIe 3.0 x8 is fine: the card only uses 8 lanes and this workload barely uses the bus.
- Linux: Nvidia driver 570+ and a PyTorch build for CUDA 12.8+ (needed for Blackwell cards).
- Bonuses: NVENC for reel renders (replaces VA-API, and avoids that driver's hang encoding 960×540 previews); interactive features (snap-to-player, re-running one clip) become instant.
- Price: $429 list; mid-2026 street prices ~$590–710 (check current).
- 16 GB is headroom, not a requirement; these models fit in 8 GB. A used RTX 3060 12GB is a cheaper option (used price not checked).

### 3) Rent GPU time
- RTX 4090 ~$0.30–0.55/hr (Vast.ai) or $0.34–0.79/hr (RunPod), mid-2026 figures.
- ~15–30 min of GPU per game, so well under $1/game; never reaches the cost of a card.
- Upload ~2.5 GB per game (~17 min at 20 Mbps up).
- **Conflicts with the "everything runs locally, no uploading video of a minor" constraint.** Vast's cheapest hosts are individuals; data-centre options (RunPod Secure Cloud, Lambda) are better but still third-party machines. Choosing this means relaxing that constraint on purpose.
- Friction: container image, upload/download scripts, starting/stopping machines. Fine for overnight batch runs, awkward for quick trial and error; interactive features stay on the CPU.

## Recommendation (2026-10-18)
If the local-only rule stays, add a GPU: it's the only option that's both private and fast enough to iterate on ball detection, the riskiest part. Renting is cheaper and fine for batch runs if the rule is relaxed, but slower to work with day to day.

## Measurements
| Hardware | Model / imgsz / frame step | fps | Full game (h) | Notes |
|---|---|---|---|---|
| Ryzen 3 3200G | YOLO11n / 640 / every frame | 8.8 | 4.3 | Misses her in 2 of 5 bench clips; ball almost never found |
| Ryzen 3 3200G | YOLO11n / 640 / every 3rd | 8.8 | 1.4 | |
| Ryzen 3 3200G | YOLO11s / 640 / every frame | 6.3 | 6.0 | |
| Ryzen 3 3200G | YOLO11n / 1280 / every frame | 4.8 | 7.8 | Finds her in 5/5; ball in 5–46% of frames |
| Ryzen 3 3200G | YOLO11n / 1280 / every 2nd | 4.8 | 3.9 | Plausible overnight player tracking |
| Ryzen 3 3200G | YOLO11s / 1280 centre crop / every frame | 2.1 | 18 | Best ball (1–53% of frames); ONNX Runtime ~15% faster |

Full game = 75 min = 135k frames. Detection is ~85–97% of the time per frame; decode 2 ms, tracker 12–20 ms.

## Sources (prices, checked 2026-10-18)
- [GPU Poet: RTX 5060 Ti price, October 2026](https://gpupoet.com/gpu/learn/price/october-2026/nvidia-geforce-rtx-5060-ti)
- [RobotDyn: RTX 5060 Ti 16GB price dynamics](https://robotdyn.com/?p=40341)
- [Spheron: RunPod vs Vast.ai 2026](https://www.spheron.network/blog/runpod-vs-vastai-2026/)
- [Costbench: Vast.ai pricing 2026](https://costbench.com/software/ai-gpu-cloud/vast-ai/)
- [Hackceleration: RunPod pricing](https://www.hackceleration.com/labs/runpod-pricing)

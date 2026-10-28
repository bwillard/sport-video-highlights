# Hardware for ML (tracking, ball, identity)

**Status:** Decided (2026-10-19): rent RunPod GPUs per job, each launch with your approval, with your OK for each kind of upload (bench clips, training data, then the game videos on a network volume). Fine-tuning and whole-game ball detection run there (about $0.50 a training run, $1.30 a game on a 5090); the filter, fixed-object finding and labelling run on this CPU. Adding a local GPU is still the option if the local-only rule should come back. Earlier status (2026-10-18): undecided, after the CPU and L4 benchmarks below.
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

Updated after the L4 test: the quality gap is now measured, not guessed. The bigger model at full resolution keeps her track for most or all of a clip (vs ~10–50% for what the CPU can run), and the bigger model finds the ball more often. Both need a GPU. An L4-class card does a whole game in ~1–3 h without any optimisation.

## Cloud jobs (2026-10-19)
`soccer-clips gpu run <job>` drives RunPod's REST API from this machine: create a Secure Cloud pod (`runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404`; RTX 4090, else 5090 / L40S), upload exactly the listed files, run, download, wipe and delete the pod in a `finally`. A dead-man's switch on the pod deletes the pod through RunPod's API after `--max-hours` (falling back to stopping its container), so billing ends even if this machine dies; for that one script the API key is sent to the pod over ssh into a root-only file (never on a command line or in the uploaded code), and it goes when the pod is deleted. The server also deletes any pod past its deadline every 10 minutes (`soccer-clips gpu sweep` by hand). Since 2026-10-27 the retrain buttons in the app (roster page, ball labels page) start jobs too: pressing one is your approval for that run. A Claude Code "ask" rule (`.claude/settings.local.json`) makes every launch need your approval. Bigger cards were judged not worth it for this job size (2026-10-19): the dataloader, not the GPU, likely limits a 700-image training run.

## Network volume (2026-10-22)
Uploading full game videos was most of a whole-game run's cost (14 GB took 68 min, ~$1 of GPU time). A 50 GB RunPod network volume in US-KS-2 (~$3.50/month) now holds the videos and the ball model; files go up over RunPod's S3-compatible API (no pod at all), and each GPU run uploads only what's new before its pod starts. A volume pins pods to its data centre, where 4090/5090 stock is thin, so runs accept any of 4090 / 5090 / L40S / RTX 6000 Ada / L40 / A6000 there and fall back to another US data centre (with a full upload) after 20 minutes. RunPod's "global networking" doesn't help: it's pod-to-pod networking capped at 100 Mbit/s, GPU pods only, not shared storage. The footage stays on RunPod until `soccer-clips gpu volume delete --yes`.

**First run on the volume (2026-10-23):** five whole games (~708k frames) on a Kansas A100 80GB PCIe ($1.79/h; nothing cheaper was free there), six workers, full precision: 227 min, ~$6.77, ~53 frames/s with the GPU at 100%. Upload: 216 KB (everything else came from the volume). Full-precision YOLO26x is what makes a whole game ~45 min here; a 4090/5090 when free, or half precision (roughly 2x on these cards), would cut that.

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

**Cloud GPU (2026-10-18):** RunPod Secure Cloud, NVIDIA L4 (24 GB; roughly 5060 Ti class), PyTorch 2.8 + CUDA, no TensorRT or batching. Same code, same results as on the CPU for the same variant (checked: identical detections), only faster. ~45 min of pod time, under $1. Footage deleted from the pod afterwards.

| Variant | fps (whole pipeline) | GPU detect ms/frame | Full game (h) | Notes |
|---|---|---|---|---|
| n640 | 49.6 | 6.5 | 0.8 | |
| n1280 | 47.4 | 7.9 | 0.8 | |
| s1280 | 43.0 | 9.9 | 0.9 | |
| s1280-centre | 63.5 | 7.3 | 0.6 | |
| s1920 (full frame, native) | 32.4 | 18.4 | 1.2 | |
| m1280-centre | 40.2 | 16.4 | 0.9 | Best ball variant |
| m1920 (full frame, native) | 19.1 | 41.0 | 2.0 | Best player tracking by far |

- At the small sizes the GPU is mostly idle: the per-frame CPU work (BoT-SORT 7–14 ms on the pod's server CPU, preprocessing) sets the pace. On the 3200G that CPU work is slower (12–20 ms), so expect ~30–50 fps at best for the small variants with this CPU, unless the tracker's camera-motion step runs on a smaller image.
- Untried speed-ups: TensorRT FP16 and batching frames, typically 2–3× on detection.
- On the 3200G, m1920 would run at an estimated ~0.3–0.4 fps (s1280 is 2.1; m is ~2.5× s, 1920 is ~2.25× 1280): days per game.

## Sources (prices, checked 2026-10-18)
- [GPU Poet: RTX 5060 Ti price, October 2026](https://gpupoet.com/gpu/learn/price/october-2026/nvidia-geforce-rtx-5060-ti)
- [RobotDyn: RTX 5060 Ti 16GB price dynamics](https://robotdyn.com/?p=40341)
- [Spheron: RunPod vs Vast.ai 2026](https://www.spheron.network/blog/runpod-vs-vastai-2026/)
- [Costbench: Vast.ai pricing 2026](https://costbench.com/software/ai-gpu-cloud/vast-ai/)
- [Hackceleration: RunPod pricing](https://www.hackceleration.com/labs/runpod-pricing)

"""
Video encoder for reel renders: the GPU's H.264 encoder (VA-API) when it's
available, otherwise x264 on the CPU.

Every piece of a render (clips, title cards, callout freezes, captions) is
encoded once, at the final quality, with the same encoder and settings, so
the final join can copy the video instead of encoding it again. Pieces made
with different settings can't be joined that way, so the render cache key
includes Encoder.key.

Full-size renders follow .env (SOCCER_ENCODER, SOCCER_GPU_QP, SOCCER_CPU_CRF,
SOCCER_CPU_SPEED; see .env.example). Previews always use fast x264: the
Radeon Vega (Raven) driver hangs encoding some sizes, 960x540 among them,
and x264 "ultrafast" is already quick at that size.
"""

import logging
import subprocess
from dataclasses import dataclass
from functools import cache

from soccer_clips.config import Settings

logger = logging.getLogger(__name__)

PREVIEW_CRF = 26
PREVIEW_SPEED = "ultrafast"
GPU_CHECK_TIMEOUT_S = 15  # the check encode takes ~1s; a hung driver never returns

# Pin the colour tags (bt709, limited range, like Veo's) so every piece has the
# same stream parameters; stream-copied pieces all play with the first piece's.
# Written into the stream rather than set with -colorspace etc., which make
# ffmpeg convert untagged sources on the CPU (impossible with GPU frames).
COLOUR_ARGS = ["-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1"
               ":matrix_coefficients=1:video_full_range_flag=0"]
# Converting an RGB picture (title card, callout frame) to video: same colours as above.
FROM_RGB = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"


@dataclass(frozen=True)
class Encoder:
    gpu_device: str | None  # None = x264 on the CPU
    quality: int  # GPU: qp; CPU: crf. Lower = better picture, bigger file.
    cpu_speed: str = "medium"

    @property
    def gpu(self) -> bool:
        return self.gpu_device is not None

    @property
    def key(self) -> list:
        return ["gpu", self.quality] if self.gpu else ["cpu", self.quality, self.cpu_speed]

    @property
    def label(self) -> str:
        return f"GPU qp {self.quality}" if self.gpu else f"CPU crf {self.quality} {self.cpu_speed}"

    def global_args(self) -> list[str]:
        """Before the inputs: the GPU device that uploaded frames go to."""
        return ["-vaapi_device", self.gpu_device] if self.gpu else []

    def gpu_decode_args(self) -> list[str]:
        """Before an input, to decode it on the GPU and keep the frames there (GPU only)."""
        return ["-hwaccel", "vaapi", "-hwaccel_device", self.gpu_device, "-hwaccel_output_format", "vaapi"]

    @property
    def upload(self) -> str:
        """Appended to a filter chain that ends in yuv420p frames on the CPU."""
        return ",format=nv12,hwupload" if self.gpu else ""

    def video_args(self) -> list[str]:
        codec = (["-c:v", "h264_vaapi", "-qp", str(self.quality)] if self.gpu
                 else ["-c:v", "libx264", "-preset", self.cpu_speed, "-crf", str(self.quality), "-pix_fmt", "yuv420p"])
        return codec + COLOUR_ARGS


@cache
def gpu_works(ffmpeg: str, device: str, width: int, height: int) -> bool:
    """Whether the GPU can encode H.264 at this size (checked once per server run)."""
    cmd = [ffmpeg, "-v", "error", "-vaapi_device", device,
           "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=30000/1001", "-frames:v", "10",
           "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-f", "null", "-"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=GPU_CHECK_TIMEOUT_S)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("GPU encoding check on %s timed out at %dx%d", device, width, height)
    except subprocess.CalledProcessError as e:
        logger.warning("GPU encoding not available on %s: %s", device, e.stderr.strip()[-300:])
    return False


def choose_encoder(settings: Settings, width: int, height: int, preview: bool) -> Encoder:
    if preview:
        return Encoder(None, PREVIEW_CRF, PREVIEW_SPEED)
    cpu = Encoder(None, settings.cpu_crf, settings.cpu_speed)
    if settings.encoder == "cpu":
        return cpu
    if gpu_works(settings.ffmpeg, settings.gpu_device, width, height):
        return Encoder(settings.gpu_device, settings.gpu_qp)
    if settings.encoder == "gpu":
        raise RuntimeError(f"SOCCER_ENCODER=gpu but the GPU ({settings.gpu_device}) can't encode "
                           f"{width}x{height}; check the server log, or set SOCCER_ENCODER=auto")
    return cpu

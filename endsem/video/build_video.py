"""Assemble the endsem video: each slide is shown for exactly as long as its narration.

Audio input, in audio/ (either form, or a mix):
  * one take per speaker, e.g. audio/aayaan.mp3, with a clear pause (>= ~1 s) between
    that speaker's slides; it is split automatically on those pauses, or
  * one clip per slide: audio/slide_01.mp3, audio/slide_02.wav, ...

Which speaker reads which slides is set in speakers.json, e.g.
  {"aayaan": [1, 2, 3], "shlok": [4, 5, 6, 7], "prapti": [8, 9, 10]}

    python3 build_video.py            # -> out/endsem_video.mp4
    python3 build_video.py --check    # only split + report clip lengths

Slides come from ../deck/png/slide_NN.png (run ../deck/render_slides.sh first).
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PNG = HERE.parent / "deck" / "png"
AUDIO, OUT = HERE / "audio", HERE / "out"
LEAD, TAIL = 0.5, 0.7          # seconds of silence before / after each clip
SILENCE_DB, SILENCE_MIN = -35, 0.9
N_SLIDES = 10


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(r.stdout.strip())


def split_take(path, n_expected, tmp):
    """Split a take on pauses into n_expected clips (keeps the n_expected longest gaps)."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                          f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    total = duration(path)
    gaps = [(e - s, s, e) for s, e in zip(starts, ends) if s > 0.05 and e < total - 0.05]
    if len(gaps) < n_expected - 1:
        sys.exit(f"{path.name}: found {len(gaps) + 1} parts, expected {n_expected}. "
                 f"Make the pauses between slides longer (>= {SILENCE_MIN}s).")
    cuts = sorted(sorted(gaps, reverse=True)[:n_expected - 1], key=lambda g: g[1])
    bounds, t0 = [], 0.0
    for _, s, e in cuts:
        bounds.append((t0, s)); t0 = e
    bounds.append((t0, total))
    clips = []
    for i, (a, b) in enumerate(bounds):
        out = tmp / f"{path.stem}_{i}.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
             "-af", f"silenceremove=start_periods=1:start_threshold={SILENCE_DB}dB,"
                    f"areverse,silenceremove=start_periods=1:start_threshold={SILENCE_DB}dB,areverse",
             "-ar", "44100", "-ac", "1", str(out)])
        clips.append(out)
    return clips


def collect_clips(tmp):
    speakers = json.loads((HERE / "speakers.json").read_text()) if (HERE / "speakers.json").exists() else {}
    clips = {}
    for name, slides in speakers.items():
        take = next((p for p in AUDIO.glob(f"{name}.*")), None)
        if take:
            for n, c in zip(slides, split_take(take, len(slides), tmp)):
                clips[n] = c
    for p in AUDIO.glob("slide_*.*"):
        clips[int(re.search(r"slide_(\d+)", p.stem).group(1))] = p
    missing = [n for n in range(1, N_SLIDES + 1) if n not in clips]
    if missing:
        sys.exit(f"no narration for slide(s) {missing}")
    return clips


def main(check_only=False):
    OUT.mkdir(exist_ok=True)
    tmp = OUT / "tmp"; tmp.mkdir(exist_ok=True)
    clips = collect_clips(tmp)
    total = 0.0
    for n in range(1, N_SLIDES + 1):
        d = duration(clips[n]); total += d + LEAD + TAIL
        print(f"slide {n:2d}: {d:5.1f} s  ({clips[n].name})")
    print(f"total ≈ {total / 60:.1f} min")
    if check_only:
        return
    segs = []
    for n in range(1, N_SLIDES + 1):
        dur = duration(clips[n]) + LEAD + TAIL
        seg = tmp / f"seg_{n:02d}.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30",
             "-i", str(PNG / f"slide_{n:02d}.png"), "-i", str(clips[n]),
             "-af", f"adelay={int(LEAD * 1000)}:all=1,apad", "-t", f"{dur:.3f}",
             "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "30",
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(seg)])
        segs.append(seg)
    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    final = OUT / "endsem_video.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", str(final)])
    print(f"wrote {final} ({duration(final) / 60:.1f} min)")


if __name__ == "__main__":
    main("--check" in sys.argv)

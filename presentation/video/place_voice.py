"""Places generated or recorded voice clips at their exact times in the video.

Input is either one long take per speaker, with a clear pause (at least about
1 s) between lines, which is split automatically on those pauses, or one
clip per line named by its ID (S1.mp3, S2.mp3, ...).

Each line is placed at its slot start. A line that runs longer than its slot
is first allowed to start up to 0.6 s early (when the previous line has
finished), then gently sped up (at most 1.15x, pitch preserved). Anything
still too long is reported, so you know which line to shorten or regenerate.

    python3 place_voice.py S path/to/shlok_full.mp3            # one take, auto-split
    python3 place_voice.py S path/to/folder_with_S1..S13/      # one file per line
    python3 place_voice.py --mux                               # add every track made so far to the video

Outputs: out/voice_<speaker>.wav (full video length) and, with --mux,
out/inside_with_voice.mp4.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
VIDEO = OUT / "inside_explainer.mp4"
MAX_TEMPO, EARLY, GUARD = 1.15, 0.6, 0.25


def ts(s):
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def video_len():
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(VIDEO)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def slots(spk):
    """[(id, start, latest_end)] for one speaker. voice_slots.json overrides SCRIPT.md."""
    override = HERE / "voice_slots.json"
    if override.exists():
        data = json.loads(override.read_text())
        if spk in data:
            return [(d["id"], ts(d["start"]), ts(d["latest_end"])) for d in data[spk]]
    pat = re.compile(r'^\|\s*([PAS]\d+)\s*\|\s*([\d:.]+)\s*\|\s*([\d:.]+)\s*\|')
    rows = [(m[1], ts(m[2])) for line in (HERE / "SCRIPT.md").read_text().splitlines() if (m := pat.match(line))]
    total = video_len()
    out = []
    for i, (lid, start) in enumerate(rows):
        nxt = rows[i + 1][1] if i + 1 < len(rows) else total
        if lid[0] == spk:
            out.append((lid, start, nxt - GUARD))
    return out


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def split_take(path, n, workdir):
    """Cut one long take into n clips on its n-1 longest pauses."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "silencedetect=noise=-38dB:d=0.45", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    total = duration(path)
    sil = [(a, b) for a, b in zip(starts, ends) if a > 0.05 and b < total - 0.05]
    if len(sil) < n - 1:
        sys.exit(f"Found only {len(sil) + 1} parts in {path.name}, expected {n}. "
                 f"Leave a clear pause (about 1.5 s) between lines and try again.")
    cuts = sorted(sorted(sil, key=lambda s: s[1] - s[0], reverse=True)[:n - 1])
    bounds, prev = [], 0.0
    for a, b in cuts:
        bounds.append((prev, a + 0.08))
        prev = b - 0.08
    bounds.append((prev, total))
    workdir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (a, b) in enumerate(bounds):
        f = workdir / f"part{i + 1:02d}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                        "-af", "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                               "silenceremove=start_periods=1:start_threshold=-45dB,areverse",
                        "-ar", "48000", "-ac", "1", str(f)], check=True)
        files.append(f)
    return files


def place(spk, src):
    sl = slots(spk)
    src = Path(src)
    if src.is_dir():
        files = []
        for lid, _, _ in sl:
            hit = sorted(src.glob(f"{lid}.*")) or sorted(src.glob(f"{lid}_*"))
            if not hit:
                sys.exit(f"missing clip for {lid} in {src}")
            files.append(hit[0])
    else:
        files = split_take(src, len(sl), OUT / f"parts_{spk}")

    total = video_len()
    inputs, chains, report, prev_end = [], [], [], 0.0
    for k, ((lid, start, latest), f) in enumerate(zip(sl, files)):
        d = duration(f)
        begin = start
        if begin + d > latest:
            begin = max(prev_end + GUARD, start - EARLY)
        tempo = 1.0
        if begin + d > latest:
            tempo = min(MAX_TEMPO, d / max(0.1, latest - begin))
        spoken = d / tempo
        end = begin + spoken
        flag = "  <-- still too long: shorten or regenerate faster" if end > latest + 0.05 else ""
        report.append(f"{lid:>4}  at {begin:7.2f}s  len {d:5.2f}s  speed x{tempo:.2f}  ends {end:7.2f}s  (limit {latest:7.2f}s){flag}")
        prev_end = end
        inputs += ["-i", str(f)]
        ms = int(round(begin * 1000))
        chain = f"[{k}:a]aresample=48000,aformat=channel_layouts=mono"
        if tempo > 1.001:
            chain += f",atempo={tempo:.4f}"
        chains.append(chain + f",adelay={ms}|{ms}[c{k}]")
    mix = "".join(f"[c{k}]" for k in range(len(files)))
    fc = ";".join(chains) + f";{mix}amix=inputs={len(files)}:normalize=0,apad,atrim=0:{total:.3f}," \
                            f"loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    out = OUT / f"voice_{spk}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc, "-map", "[out]",
                    "-ar", "48000", "-ac", "1", str(out)], check=True)
    print("\n".join(report))
    print("wrote", out)


def mux():
    tracks = sorted(OUT.glob("voice_*.wav"))
    if not tracks:
        sys.exit("no voice_*.wav tracks yet")
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(VIDEO)]
    for t in tracks:
        args += ["-i", str(t)]
    n = len(tracks)
    fc = "".join(f"[{i + 1}:a]" for i in range(n)) + f"amix=inputs={n}:normalize=0,alimiter=limit=0.95[a]"
    out = OUT / "inside_with_voice.mp4"
    subprocess.run(args + ["-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)
    print("tracks:", ", ".join(t.name for t in tracks))
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--mux"]:
        mux()
    elif len(sys.argv) == 3:
        place(sys.argv[1].upper(), sys.argv[2])
    else:
        sys.exit(__doc__)

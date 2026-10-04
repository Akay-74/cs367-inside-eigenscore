"""Builds the voiceover guide video: the original silent explainer with the
script burned in as timed, speaker-coloured subtitles.

Dialogue, line IDs and start/end times are read straight from SCRIPT.md, so the
guide always matches the cue sheet. The original 1920x1080 frame is kept
intact; a band is added above (speaker name) and below (subtitles) so nothing
in the animation is covered.

    python3 make_guide.py      -> out/inside_voiceover_guide.mp4
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "out" / "inside_explainer.mp4"
OUT = HERE / "out" / "inside_voiceover_guide.mp4"
ASS = HERE / "out" / "voiceover_guide.ass"

TOP, VID_H, BOTTOM, W = 110, 1080, 230, 1920
H = TOP + VID_H + BOTTOM

# speaker -> (display name, ASS colour &HBBGGRR&)
SPEAKERS = {
    "P": ("PRAPTI", "&H00B18FFF&"),   # pink   #FF8FB1
    "A": ("AAYAAN", "&H00FFC47C&"),   # blue   #7CC4FF
    "S": ("SHLOK", "&H0057C8FF&"),    # amber  #FFC857
}
DIM = "&H00808A99&"


def ts(s):
    """'1:04.5' -> seconds"""
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def ass_time(t):
    t = max(0.0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def parse_script():
    rows = []
    pat = re.compile(r'^\|\s*([PAS]\d+)\s*\|\s*([\d:.]+)\s*\|\s*([\d:.]+)\s*\|\s*(.*?)\s*\|\s*"(.*)"\s*\|\s*$')
    for line in (HERE / "SCRIPT.md").read_text().splitlines():
        m = pat.match(line)
        if m:
            rows.append(dict(id=m[1], spk=m[1][0], start=ts(m[2]), end=ts(m[3]), text=m[5]))
    return rows


def phrases(text, limit=62):
    """Split a line into short readable phrases: sentences first, then clauses."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= limit:
            out.append(sent)
            continue
        parts = re.split(r"(?<=[,:;])\s+", sent)
        cur = ""
        for p in parts:
            if cur and len(cur) + 1 + len(p) > limit:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
        if cur:
            out.append(cur)
    # glue very short fragments onto the previous phrase
    merged = []
    for p in out:
        if merged and len(p) < 14 and len(merged[-1]) + len(p) < limit + 12:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def esc(s):
    return s.replace("{", "(").replace("}", ")")


def build_ass(rows):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Liberation Sans,58,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,-1,0,0,0,100,100,0,0,1,3,0,2,90,90,0,1
Style: Tag,Liberation Sans,46,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,-1,0,0,0,100,100,3,0,1,0,0,5,0,0,0,1
Style: Small,Liberation Sans,30,&H00808A99&,&H00808A99&,&H00000000&,&H00000000&,0,0,0,0,100,100,1,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    tag_y, sub_y = TOP // 2, TOP + VID_H + BOTTOM // 2

    def add(layer, a, b, style, text):
        ev.append(f"Dialogue: {layer},{ass_time(a)},{ass_time(b)},{style},,0,0,0,,{text}")

    # speaker blocks (contiguous runs of one speaker)
    blocks = []
    for r in rows:
        if blocks and blocks[-1]["spk"] == r["spk"]:
            blocks[-1]["end"] = r["end"]
        else:
            blocks.append(dict(spk=r["spk"], start=r["start"], end=r["end"]))

    for i, b in enumerate(blocks):
        name, col = SPEAKERS[b["spk"]]
        # whose turn it is during pauses inside the block: dimmed name, no dialogue
        mine = [r for r in rows if r["spk"] == b["spk"] and b["start"] <= r["start"] <= b["end"]]
        edges = [b["start"] - 0.6] + [x for r in mine for x in (r["start"], r["end"])] + [b["end"] + 0.4]
        for a, z in zip(edges[0::2], edges[1::2]):
            if z - a > 0.05:
                add(0, a, z, "Tag", f"{{\\an5\\pos({W//2},{tag_y})\\c{DIM}}}{name}'S TURN  ·  pause")
        # "up next" hint in the 3 s before a new speaker's first line
        if i + 1 < len(blocks):
            nxt = blocks[i + 1]
            n_name, n_col = SPEAKERS[nxt["spk"]]
            add(2, max(b["end"] + 0.4, nxt["start"] - 3.0), nxt["start"] - 0.6, "Small",
                f"{{\\an5\\pos({W//2},{tag_y})}}up next:  {{\\c{n_col}\\b1}}{n_name}")

    for r in rows:
        name, col = SPEAKERS[r["spk"]]
        # bright speaker label while the line is being spoken (covers the dim one)
        add(1, r["start"], r["end"], "Tag",
            f"{{\\an5\\pos({W//2},{tag_y})\\c{col}}}● {name} SPEAKING  {{\\c&H00B0B8C4&\\fs30}}{r['id']}")
        # subtitle phrases, timed in proportion to their length
        ps = phrases(r["text"])
        total = sum(len(p) for p in ps)
        dur = r["end"] - r["start"]
        t = r["start"]
        for k, p in enumerate(ps):
            seg = dur * len(p) / total
            a, b = t, (r["end"] if k == len(ps) - 1 else t + seg)
            add(3, a, b, "Sub", f"{{\\an5\\pos({W//2},{sub_y})\\c{col}}}{esc(p)}")
            t += seg
    ASS.write_text(head + "\n".join(ev) + "\n")


def render():
    vf = (f"pad={W}:{H}:0:{TOP}:color=0x0F1A2B,"
          f"drawbox=x=0:y={TOP-3}:w={W}:h=3:color=0x2A3B57:t=fill,"
          f"drawbox=x=0:y={TOP+VID_H}:w={W}:h=3:color=0x2A3B57:t=fill,"
          f"subtitles={ASS.name}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SRC.name, "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", OUT.name], cwd=SRC.parent, check=True)


if __name__ == "__main__":
    rows = parse_script()
    assert len(rows) == 38, f"expected 38 script lines, parsed {len(rows)}"
    build_ass(rows)
    render()
    print("wrote", OUT)

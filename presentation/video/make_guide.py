"""Builds the voiceover guide video: the original silent explainer with the
script burned in as karaoke-style, speaker-coloured subtitles.

How a line looks to the reader:
  * GET READY: the phrase appears dimmed (up to 1.5 s) before it must be spoken,
    and the status row says "<NAME> · get ready".
  * SPEAK NOW: the phrase fills with the speaker's colour from left to right,
    word by word, at the pace it should be spoken.
  * The row underneath previews the next phrase, so there is always something
    to read ahead.
Everything (who is speaking, whose turn, up next, the text) sits in one band
below the untouched 1920x1080 video, so you never look up and down.

Dialogue, line IDs and start/end times are read straight from SCRIPT.md.

    python3 make_guide.py      -> out/inside_voiceover_guide.mp4
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "out" / "inside_explainer.mp4"
OUT = HERE / "out" / "inside_voiceover_guide.mp4"
ASS = HERE / "out" / "voiceover_guide.ass"

W, VID_H, BAND = 1920, 1080, 300
H = VID_H + BAND
Y_STATUS, Y_MAIN, Y_NEXT = VID_H + 52, VID_H + 150, VID_H + 245
LEAD = 1.5  # max seconds a phrase is shown before it must be spoken

# speaker -> (display name, ASS colour &HBBGGRR&)
SPEAKERS = {
    "P": ("PRAPTI", "&H00B18FFF&"),   # pink   #FF8FB1
    "A": ("AAYAAN", "&H00FFC47C&"),   # blue   #7CC4FF
    "S": ("SHLOK", "&H0057C8FF&"),    # amber  #FFC857
}
DIM = "&H00808A99&"        # grey, not yet spoken
DIMMER = "&H00606A79&"


def ts(s):
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
        cur = ""
        for p in re.split(r"(?<=[,:;])\s+", sent):
            if cur and len(cur) + 1 + len(p) > limit:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
        if cur:
            out.append(cur)
    merged = []
    for p in out:
        if merged and len(p) < 14 and len(merged[-1]) + len(p) < limit + 12:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def esc(s):
    return s.replace("{", "(").replace("}", ")")


def all_phrases(rows):
    """Every phrase with its own spoken window, timed in proportion to length."""
    out = []
    for r in rows:
        ps = phrases(r["text"])
        total = sum(len(p) for p in ps)
        t = r["start"]
        for k, p in enumerate(ps):
            end = r["end"] if k == len(ps) - 1 else t + (r["end"] - r["start"]) * len(p) / total
            out.append(dict(row=r, spk=r["spk"], text=p, start=t, end=end, first=(k == 0)))
            t = end
    for i, ph in enumerate(out):
        prev_end = out[i - 1]["end"] if i else 0.0
        ph["show"] = max(ph["start"] - LEAD, prev_end)
    return out


def karaoke(ph, col):
    """Dim text that fills with the speaker colour word by word while spoken."""
    words = ph["text"].split(" ")
    weights = [len(w) + 1 for w in words]
    total_cs = round((ph["end"] - ph["start"]) * 100)
    lead_cs = round((ph["start"] - ph["show"]) * 100)
    parts, used = [f"{{\\k{lead_cs}}}"], 0
    for i, (w, wt) in enumerate(zip(words, weights)):
        cs = total_cs - used if i == len(words) - 1 else round(total_cs * wt / sum(weights))
        used += cs
        parts.append(f"{{\\kf{cs}}}{esc(w)}" + ("" if i == len(words) - 1 else " "))
    return f"{{\\an5\\pos({W//2},{Y_MAIN})\\1c{col}\\2c{DIM}}}" + "".join(parts)


def status_timeline(rows, phs):
    """Non-overlapping status segments: speak now > get ready > up next > pause."""
    cand = []  # (priority, start, end, text)
    for r in rows:
        name, col = SPEAKERS[r["spk"]]
        cand.append((4, r["start"], r["end"],
                     f"{{\\c{col}}}●  {name}  ·  SPEAK NOW   {{\\c&H00B0B8C4&\\fs28}}{r['id']}"))
    for ph in phs:
        if ph["first"] and ph["start"] - ph["show"] > 0.05:
            name, col = SPEAKERS[ph["spk"]]
            cand.append((3, ph["show"], ph["start"],
                         f"{{\\c{DIM}}}○  {{\\c{col}}}{name}{{\\c{DIM}}}  ·  get ready…   {{\\fs28}}{ph['row']['id']}"))
    blocks = []
    for r in rows:
        if blocks and blocks[-1]["spk"] == r["spk"]:
            blocks[-1]["end"] = r["end"]
        else:
            blocks.append(dict(spk=r["spk"], start=r["start"], end=r["end"]))
    for i, b in enumerate(blocks):
        name, _ = SPEAKERS[b["spk"]]
        cand.append((1, b["start"] - LEAD - 0.5, b["end"] + 0.4,
                     f"{{\\c{DIMMER}}}{name}'S TURN  ·  pause"))
        if i + 1 < len(blocks):
            nn, ncol = SPEAKERS[blocks[i + 1]["spk"]]
            cand.append((2, b["end"] + 0.4, blocks[i + 1]["start"] - LEAD,
                         f"{{\\c{DIM}}}up next:  {{\\c{ncol}}}{nn}"))
    cuts = sorted({max(0.0, x) for c in cand for x in (c[1], c[2])})
    segs = []
    for a, z in zip(cuts, cuts[1:]):
        live = [c for c in cand if c[1] <= a + 1e-6 and c[2] >= z - 1e-6]
        if not live:
            continue
        txt = max(live, key=lambda c: c[0])[3]
        if segs and segs[-1][2] == txt and abs(segs[-1][1] - a) < 1e-6:
            segs[-1][1] = z
        else:
            segs.append([a, z, txt])
    return segs


def build_ass(rows):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Liberation Sans,56,&H00FFFFFF&,{DIM},&H00000000&,&H00000000&,-1,0,0,0,100,100,0,0,1,2,0,5,40,40,0,1
Style: Status,Liberation Sans,38,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,-1,0,0,0,100,100,2,0,1,0,0,5,0,0,0,1
Style: Next,Liberation Sans,34,{DIMMER},{DIMMER},&H00000000&,&H00000000&,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []

    def add(layer, a, b, style, text):
        if b - a > 0.02:
            ev.append(f"Dialogue: {layer},{ass_time(a)},{ass_time(b)},{style},,0,0,0,,{text}")

    phs = all_phrases(rows)
    for i, ph in enumerate(phs):
        _, col = SPEAKERS[ph["spk"]]
        add(1, ph["show"], ph["end"], "Main", karaoke(ph, col))
        # read-ahead: the phrase after this one, if it comes soon
        if i + 1 < len(phs) and phs[i + 1]["start"] - ph["end"] < 8:
            nx = phs[i + 1]
            label = "next:"
            if nx["spk"] != ph["spk"]:
                nn, ncol = SPEAKERS[nx["spk"]]
                label = f"next ({{\\c{ncol}}}{nn}{{\\c{DIMMER}}}):"
            add(1, ph["show"], nx["show"], "Next",
                f"{{\\an5\\pos({W//2},{Y_NEXT})}}{label}  {esc(nx['text'])}")
    for a, z, txt in status_timeline(rows, phs):
        add(0, a, z, "Status", f"{{\\an5\\pos({W//2},{Y_STATUS})}}{txt}")
    ASS.write_text(head + "\n".join(ev) + "\n")


def render():
    vf = (f"pad={W}:{H}:0:0:color=0x0F1A2B,"
          f"drawbox=x=0:y={VID_H}:w={W}:h=3:color=0x2A3B57:t=fill,"
          f"drawbox=x=560:y={VID_H+95}:w=800:h=1:color=0x2A3B57:t=fill,"
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

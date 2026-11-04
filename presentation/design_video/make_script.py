"""Writes VOICEOVER.md (ElevenLabs Multilingual v2 scripts with <break> tags) from lines.py."""
import lines
L, SC = lines.timeline()
TL = {k: v for k, v in L.items()}
ORDER = [i for i, _, _ in lines.LINES]
TEXT = {i: t for i, _, t in lines.LINES}
SCENE = {i: s for i, s, _ in lines.LINES}
SCN = {"s1": "Title", "s2": "1 · Problem", "s3": "2 · The idea", "s4": "3 · Eigenvalues", "s5": "4 · Pipeline",
       "s6": "5 · 8 GB GPU", "s7": "6 · Results", "s8": "7 · Future directions", "s9": "Thank you"}
TONE = {
 "P1": "warm, welcoming; small pause after “Hello everyone.”",
 "P2": "serious, setting up the problem",
 "P3": "clear and purposeful; slight stress on “without an answer key”",
 "P4": "lighter, inviting",
 "P5": "contrast: steady on “agree”, a little lift on “scatter”",
 "P6": "firm; stress “internal states”",
 "A1": "explanatory, even pace",
 "A2": "calm, matter-of-fact",
 "A3": "slightly stronger; land on “flagged”",
 "A4": "technical, confident; clear on “sixteen of thirty two”",
 "A5": "steady listing pace, do not rush the numbers",
 "A6": "practical; small lift on “three point nine”",
 "S1": "quick list of three things",
 "S2": "factual",
 "S3": "confident; a small lift on “eighty three point five”",
 "S4": "forward-looking",
 "S5": "wrap-up tone, falling at the end",
 "S6": "warm, closing",
}
NAMES = {"P": "Prapti", "A": "Aayaan", "S": "Shlok"}

def mmss(t):
    return f"{int(t//60)}:{t%60:04.1f}"

def block(spk):
    ids = [i for i in ORDER if i[0] == spk]
    out = []
    for k, i in enumerate(ids):
        txt = TEXT[i]
        if k < len(ids) - 1:
            gap = TL[ids[k+1]][0] - TL[i][1]
            gap = max(0.6, round(gap, 1))
            # ElevenLabs allows at most 3 s per break tag
            tags = []
            while gap > 3.0:
                tags.append('<break time="3.0s" />'); gap = round(gap - 3.0, 1)
            tags.append(f'<break time="{gap:.1f}s" />')
            out.append(txt + " " + " ".join(tags))
        else:
            out.append(txt)
    return "\n".join(out), ids

md = ["# Design-project video: voiceover scripts (ElevenLabs v2)\n",
      "Video: `design_video_silent.mp4`, 2:00, 1920×1080. Order: **Prapti → Aayaan → Shlok**.\n",
      "Each person generates **one audio file** from their block below and sends it back. The `<break>` tags already hold the right gaps, so each file plays in sync if it starts at that person’s first cue. When merging, I place every line on its cue and nudge a line by a few percent only if it runs long.\n",
      "## ElevenLabs settings (same for all three)\n",
      "| Setting | Value |\n|---|---|\n| Model | **Eleven Multilingual v2** |\n| Voice | your own cloned voice |\n| Speed | **1.0** (if your file is longer than the target below, re-generate at 1.05–1.1) |\n| Stability | 50% |\n| Similarity | 80% |\n| Style exaggeration | 0% |\n| Speaker boost | on |\n| Output | MP3 44.1 kHz 192 kbps (or WAV) |\n",
      "Paste the block exactly as written. Numbers are spelled out on purpose so they are pronounced correctly. Don’t add or remove `<break>` tags.\n"]
for spk in "PAS":
    text, ids = block(spk)
    first, last = TL[ids[0]][0], TL[ids[-1]][1]
    md.append(f"---\n\n## {NAMES[spk]}: {mmss(first)} → {mmss(last)}  (target file length ≈ {last-first:.0f} s)\n")
    md.append("```\n" + text + "\n```\n")
    md.append("| # | Scene | Starts at | Must end by | Line | Tone |\n|---|---|---|---|---|---|")
    for k, i in enumerate(ids):
        nxt = ORDER.index(i) + 1
        end_by = (TL[ORDER[nxt]][0] - 0.2) if nxt < len(ORDER) else 119.5
        md.append(f"| {i} | {SCN[SCENE[i]]} | {mmss(TL[i][0])} | {mmss(end_by)} | {TEXT[i]} | {TONE[i]} |")
    md.append("")
md.append("---\n\n## Whole timeline at a glance\n")
md.append("| Time | Scene | Speaker |\n|---|---|---|")
for s, (a, b) in SC.items():
    spk = sorted({NAMES[i[0]] for i in ORDER if SCENE[i] == s}, key="Prapti Aayaan Shlok".split().index)
    md.append(f"| {mmss(a)}–{mmss(120.0 if s=="s9" else b)} | {SCN[s]} | {' → '.join(spk)} |")
md.append("\nTip: listen once before sending. If a word is mispronounced (for example “Aayaan” or “Chotaliya”), re-generate just that take; spelling it phonetically (“Aa-yaan”, “Cho-ta-li-ya”) usually fixes it.\n")
open("VOICEOVER.md", "w").write("\n".join(md))
print("\n".join(md))

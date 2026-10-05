import json, re, subprocess
from pathlib import Path
VID = Path("/home/user/cs367-inside-eigenscore/presentation/video")
OUT = Path("out_voice"); OUT.mkdir(exist_ok=True)
TOTAL = 254.033
def ts(s): m,x=s.split(":"); return int(m)*60+float(x)
def dur(f): return float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(f)],capture_output=True,text=True).stdout)
pat = re.compile(r'^\|\s*([PAS]\d+)\s*\|\s*([\d:.]+)\s*\|')
allrows = [(m[1], ts(m[2])) for l in (VID/"SCRIPT.md").read_text().splitlines() if (m:=pat.match(l))]
def slots(spk):
    js = json.loads((VID/"voice_slots.json").read_text())
    if spk in js: return [(d["id"], ts(d["start"]), ts(d["latest_end"])) for d in js[spk]]
    out=[]
    for i,(lid,st) in enumerate(allrows):
        if lid[0]==spk: out.append((lid, st, (allrows[i+1][1] if i+1<len(allrows) else TOTAL)-0.25))
    return out
EARLY, GAP, MAXT = 0.6, 0.2, 1.15
def place(spk, clipdir, pre=""):
    sl = slots(spk); prev_end = 0; ins=[]; ch=[]; rep=[]
    for k,(lid,st,latest) in enumerate(sl):
        f = Path(clipdir)/f"{lid}.wav"; d = dur(f)
        begin = max(st, prev_end+GAP)
        if begin + d > latest:
            begin = max(prev_end+GAP, min(begin, st-EARLY) if begin<=st else begin)
        room = latest - begin
        tempo = 1.0 if d <= room else min(MAXT, d/max(room,0.1))
        end = begin + d/tempo
        rep.append(f"{lid:>4} slot {st:7.2f}  placed {begin:7.2f} ({begin-st:+.2f}s)  x{tempo:.2f}  ends {end:7.2f} / limit {latest:7.2f}{'  OVER' if end>latest+0.05 else ''}")
        prev_end = end
        ins += ["-i", str(f)]; ms=int(round(begin*1000))
        c = f"[{k}:a]aresample=48000,aformat=channel_layouts=mono" + (f",{pre}" if pre else "")
        if tempo>1.001: c += f",atempo={tempo:.4f}"
        ch.append(c+f",adelay={ms}|{ms}[c{k}]")
    mix="".join(f"[c{k}]" for k in range(len(sl)))
    fc=";".join(ch)+f";{mix}amix=inputs={len(sl)}:normalize=0,apad,atrim=0:{TOTAL},loudnorm=I=-16:TP=-1.5:LRA=11[o]"
    out=OUT/f"voice_{spk}.wav"
    subprocess.run(["ffmpeg","-y","-loglevel","error",*ins,"-filter_complex",fc,"-map","[o]","-ar","48000","-ac","1",str(out)],check=True)
    print("\n".join(rep)); return out
s = place("S","clips_S")
a = place("A","clips_A", pre="highpass=f=80,afftdn=nr=12:nf=-40")

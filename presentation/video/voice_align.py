import json, re, sys, difflib, subprocess
from pathlib import Path
VID = Path("/home/user/cs367-inside-eigenscore/presentation/video")
NUM = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine","10":"ten","12":"twelve","16":"sixteen","95":"ninetyfive","250":"twohundredfifty","3000":"threethousand","8000":"eightthousand"}
def norm(text):
    text = text.lower().replace("-", " ").replace("’","'")
    toks = re.findall(r"[a-z0-9.']+", text)
    out=[]
    for t in toks:
        t=t.strip(".'")
        if not t: continue
        out.append(NUM.get(t,t))
    return out
def script_lines(spk):
    pat = re.compile(r'^\|\s*([PAS]\d+)\s*\|\s*([\d:.]+)\s*\|\s*([\d:.]+)\s*\|\s*(.*?)\s*\|\s*"(.*)"\s*\|\s*$')
    return [(m[1], m[5]) for l in (VID/"SCRIPT.md").read_text().splitlines() if (m:=pat.match(l)) and m[1][0]==spk]
def align(spk, words_json, audio, outdir):
    words = json.load(open(words_json))
    tw = [norm(w["w"]) for w in words]
    flat, owner = [], []
    for i,t in enumerate(tw):
        for x in t: flat.append(x); owner.append(i)
    lines = script_lines(spk)
    sflat, sline = [], []
    for k,(lid,txt) in enumerate(lines):
        for x in norm(txt): sflat.append(x); sline.append(k)
    sm = difflib.SequenceMatcher(None, sflat, flat, autojunk=False)
    first, last = {}, {}
    for a,b,n in sm.get_matching_blocks():
        for j in range(n):
            k = sline[a+j]; wi = owner[b+j]
            first[k] = min(first.get(k, 1e9), wi); last[k] = max(last.get(k, -1), wi)
    dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",audio],capture_output=True,text=True).stdout)
    spans=[]
    for k,(lid,txt) in enumerate(lines):
        if k not in first: sys.exit(f"could not find {lid} in the audio")
        spans.append([words[first[k]]["s"], words[last[k]]["e"]])
    # a line ends with the last transcribed word before the next line starts (keeps unmatched tails like ".6")
    for k in range(len(spans)-1):
        nxt = first[k+1]
        spans[k][1] = max(spans[k][1], max(w["e"] for w in words[:nxt]) if nxt else spans[k][1])
    # move each boundary to the longest silence among the nearby words (robust to misheard names)
    for k in range(len(lines)-1):
        cur = first[k+1]
        gap = lambda j: words[j]["s"]-words[j-1]["e"]
        if cur <= 0 or gap(cur) >= 0.15:
            continue                      # boundary already sits in a real pause
        lo, hi = max(1, cur-1), min(len(words)-1, cur+1)
        best = max(range(lo, hi+1), key=gap)
        if gap(best) < 0.4:
            continue
        first[k+1] = best
        spans[k][1] = words[best-1]["e"]; spans[k+1][0] = words[best]["s"]
    # cut points: middle of the gap between consecutive lines
    cuts=[]
    for k in range(len(spans)):
        a = 0.0 if k==0 else (spans[k-1][1]+spans[k][0])/2
        b = dur if k==len(spans)-1 else (spans[k][1]+spans[k+1][0])/2
        a = max(a, spans[k][0]-0.25); b = min(b, spans[k][1]+0.35)
        cuts.append((a,b))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for (lid,txt),(a,b) in zip(lines,cuts):
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",audio,"-ss",f"{a:.3f}","-to",f"{b:.3f}",
                        "-af","afade=t=in:d=0.03,areverse,afade=t=in:d=0.06,areverse","-ac","1","-ar","48000",f"{outdir}/{lid}.wav"],check=True)
        print(f"{lid:>4} {a:6.2f}-{b:6.2f} ({b-a:4.2f}s)  {' '.join(w['w'].strip() for w in words if w['s']>=a-0.01 and w['e']<=b+0.01)}")
align(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

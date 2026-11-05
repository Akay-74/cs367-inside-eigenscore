"""Single source of truth for the design-project video voiceover.
Each line: (id, scene, text-as-spoken). Durations are estimated from syllables
(ElevenLabs Multilingual v2, speed 1.0 ~ 4.3 syllables/s) so scenes can be sized."""
import re
LINES = [
 ("P1","s1","Hello everyone. I'm Prapti Chotaliya, and my teammates are Aayaan Ali Khan and Shlok Bhavsar."),
 ("P2","s2","Large language models often give wrong answers, in the same confident tone as correct ones."),
 ("P3","s2","We want to flag likely wrong answers, without an answer key, on an eight gigabyte GPU."),
 ("P4","s3","Our idea is simple. We ask the model the same question ten times."),
 ("P5","s3","If it knows, the answers agree. If it is guessing, they scatter."),
 ("P6","s3","We compare them through the model's internal states, not their words."),
 ("A1","s4","Each answer becomes a vector inside the network, and we measure how spread out these vectors are."),
 ("A2","s4","A tight cluster gives small eigenvalues and a low score."),
 ("A3","s4","A wide spread gives large eigenvalues and a high score, so the answer is flagged."),
 ("A4","s5","We use Llama 2, seven billion, and read layer sixteen of thirty two, at the last token of each answer."),
 ("A5","s5","Ten answers give a ten by four thousand ninety six matrix, which reduces to ten eigenvalues, and one score."),
 ("A6","s6","To fit an eight gigabyte GPU, we load the model in four bit, from thirteen point five down to three point nine gigabytes."),
 ("S1","s6","A hook keeps only the vector we need, answers are generated two at a time, and eight unit tests check the maths."),
 ("S2","s7","On two hundred and fifty test questions, the model was wrong on one hundred and fifty eight."),
 ("S3","s7","Our detection score, AUROC, is eighty three point five percent, where random guessing gives fifty."),
 ("S4","s8","As future directions, we will combine a band of middle layers instead of one,"),
 ("S5","s8","add token confidence, and sample fewer answers when they clearly agree."),
 ("S6","s9","Thank you."),
]
def syl(w):
    w=w.lower().strip(".,'?!")
    if w in("gpu",):return 3
    if w=="auroc":return 3
    if w in("aayaan",):return 2
    if w=="llama":return 2
    v=re.findall(r"[aeiouy]+",w)
    n=len(v)
    if w.endswith("e") and n>1 and not w.endswith(("le","ee"))  :n-=1
    return max(1,n)
def est(text,rate=4.3):
    words=text.split()
    s=sum(syl(w) for w in words)
    return s/rate+0.22*text.count(",")+0.35*len(re.findall(r"[.!?](?=\s)",text))+0.1
LEAD={"s1":0.9}
# Cue moved after the real recordings came in: S4 and S5 are one sentence,
# so S5 starts right after S4 ends (S4 recording: 104.44 + 4.07 s).
OVERRIDE={"S5":108.80}
def timeline(gap=0.6,scene_gap=1.1,lead=0.7,tail=0.6):
    """Return line timings {id:(start,end)} and scene windows {scene:(start,end)}."""
    t=0.0;cur=None;L={};SC={}
    for i,sc,txt in LINES:
        if sc!=cur:
            if cur: SC[cur]=(SC[cur][0],t+tail-0.45) ; 
            start_scene=(t+tail-0.45) if cur else 0.0
            SC[sc]=(start_scene,None)
            t=start_scene+lead
            cur=sc
        d=est(txt);L[i]=(round(t,2),round(t+d,2));t+=d+gap
    SC[cur]=(SC[cur][0],t+2.0)
    for k,v in OVERRIDE.items():
        d=L[k][1]-L[k][0];L[k]=(v,round(v+d,2))
    return L,SC
if __name__=="__main__":
    tot={}
    for i,sc,t in LINES:
        d=est(t);tot[sc]=tot.get(sc,0)+d;tot[i[0]]=tot.get(i[0],0)+d
        print(f"{i} {sc} {d:5.1f}  {t}")
    print(tot, sum(v for k,v in tot.items() if k.startswith('s')))

def freeze(path="timeline.js", total=120.0):
    """Write the timeline the animation reads (window.TL)."""
    import json
    L,SC=timeline()
    sc=[{"id":k,"start":round(v[0],2),"end":round(v[1],2)} for k,v in SC.items()]
    sc[-1]["end"]=total
    lines={k:{"start":v[0],"end":v[1],"spk":k[0]} for k,v in L.items()}
    open(path,"w").write("window.TL="+json.dumps({"scenes":sc,"lines":lines,"total":total})+";\n")
    return sc,lines

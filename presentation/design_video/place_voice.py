# Places the three ElevenLabs recordings (P.mp3, A.mp3, S.mp3 in the working dir) on the
# line cues of timeline.js. Segment bounds were read with ffmpeg silencedetect.
# Writes voice.wav; mux with: ffmpeg -i out/design_video_silent.mp4 -i voice.wav -map 0:v -map 1:a -c:v copy -c:a aac design_video_final.mp4
import json,re,subprocess
TL=json.loads(re.search(r"window.TL=(.*);",open('/home/user/cs367-inside-eigenscore/presentation/design_video/timeline.js').read()).group(1))
seg={'P':[(0,5.47),(6.98,12.03),(12.82,17.73),(19.50,23.80),(24.61,29.05),(29.86,33.75)],
     'A':[(0,4.91),(5.71,8.90),(9.89,14.31),(16.16,22.22),(23.13,29.56),(31.34,38.22)],
     'S':[(0,5.96),(7.81,11.91),(12.75,18.78),(20.48,24.42),(25.21,28.58),(30.34,31.06)]}
ids={'P':['P1','P2','P3','P4','P5','P6'],'A':['A1','A2','A3','A4','A5','A6'],'S':['S1','S2','S3','S4','S5','S6']}
order=[k for k in TL['lines']]
inputs=[];filt=[];n=0
for spk in 'PAS':
    for (a,b),i in zip(seg[spk],ids[spk]):
        a=max(0,a-0.05);b=b+0.08
        st=TL['lines'][i]['start'];k=order.index(i)
        nxt=TL['lines'][order[k+1]]['start'] if k+1<len(order) else 120
        print(f"{i} dur {b-a:5.2f} at {st:6.2f} ends {st+b-a:6.2f} next {nxt:6.2f}", "OVERLAP" if st+b-a>nxt else "")
        inputs+=['-i',f'{spk}.mp3']
        filt.append(f"[{n}:a]atrim={a}:{b},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:d=0.03,afade=t=out:st={b-a-0.06}:d=0.06,adelay={int(st*1000)}|{int(st*1000)}[x{n}]")
        n+=1
fc=";".join(filt)+";"+"".join(f"[x{j}]" for j in range(n))+f"amix=inputs={n}:normalize=0,apad,atrim=0:120,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
subprocess.run(['ffmpeg','-y','-v','error',*inputs,'-filter_complex',fc,'-map','[out]','-ar','48000','voice.wav'],check=True)

# INSIDE explainer: voiceover script

Video: `out/inside_explainer.mp4` (3:42, 1920×1080, silent; record the voiceover on top).
The animation carries the details. The voiceover is about 2 minutes 20 seconds, so there are pauses where the visuals play on their own.
Start each line at the time shown. The lines are short enough to finish before the next scene.

| Time | Scene | Speaker | Voiceover |
|---|---|---|---|
| 0:01 | Title | Aayaan | "Hi, we are Aayaan, Shlok and Prapti. For CS307 we implemented INSIDE, an ICLR 2024 paper on detecting hallucinations in large language models." |
| 0:16 | The problem | Aayaan | "Language models sound equally confident when they are wrong." |
| 0:22 | | | *(pause: the answers animate in)* |
| 0:29 | | Aayaan | "But ask the same question several times: if the model knows, the answers agree. If it is guessing, they scatter." |
| 0:40 | Look inside | Aayaan | "INSIDE does not compare the words. It reads the model's internal state: the middle-layer vector of each answer's last token." |
| 0:50 | | Aayaan | "Ten answers give ten vectors." |
| 1:02 | The math | Shlok | "EigenScore measures how spread out these vectors are." |
| 1:06 | | Shlok | "We build their covariance matrix and take its eigenvalues, which are the spread along each direction. The score is the average log-eigenvalue." |
| 1:18 | | | *(pause: the formula appears)* |
| 1:20 | | Shlok | "If the answers agree, the eigenvalues collapse and the score is low." |
| 1:28 | | Shlok | "If they disagree, the eigenvalues grow and the score is high: a likely hallucination." |
| 1:43 | Blind spot | Shlok | "One blind spot: if the model is confidently wrong, every answer is the same, and the score trusts it." |
| 1:51 | | Shlok | "The paper's fix is feature clipping: cut off the extreme activations in the penultimate layer, so overconfident errors start to show." |
| 2:11 | Implementation | Prapti | "The official code did not run. We fixed twelve bugs," |
| 2:20 | | Prapti | "implemented feature clipping, which was missing, ran a 7-billion-parameter model on an 8 GB laptop GPU, and verified everything with eight tests." |
| 2:37 | Results | Prapti | "On 250 CoQA questions, the paper's ranking reproduced exactly, with EigenScore the best at 83.5 AUROC." |
| 2:51 | | Prapti | "The effect of feature clipping is too small to measure at this scale; the interval includes both zero and the paper's gain." |
| 3:07 | End-sem plan | Aayaan | "By the end-semester, we will improve the method itself." |
| 3:13 | | Aayaan | "Instead of one hand-picked layer, we combine a band of middle layers into one score," |
| 3:21 | | Aayaan | "and we add token confidence to catch confident mistakes, and adaptive sampling to cut cost." |
| 3:33 | Thank you | Aayaan | "Thank you." |

Speaking time: Aayaan about 55 s, Shlok about 45 s, Prapti about 40 s.

## Recording and editing

1. Play the MP4 and read along once to check the timing.
2. Record each speaker's lines as separate clips: a quiet room, a phone about 20 cm away.
3. In any editor (Clipchamp, CapCut, DaVinci Resolve, iMovie), place the clips at the times above. Optionally add quiet background music at about −28 dB.
4. Export as MP4, 1080p.

To change text or timing: edit `index.html`, preview it in a browser (space: pause, arrows: scene), then run `node render.mjs` (requires Node with Playwright and ffmpeg).

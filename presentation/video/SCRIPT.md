# INSIDE explainer: voiceover script and cue sheet

Video: `out/inside_explainer.mp4`, 4:14 long, no sound. Speaking order: **Prapti → Aayaan → Shlok**.

| Speaker | Slides | Video time | Speaking time |
|---|---|---|---|
| Prapti | 1 Title · 2 Problem · 3 Key idea | 0:00 – 1:00 | about 45 s |
| Aayaan | 4 Math · 5 Feature clipping · 6 Their code vs. ours | 1:00 – 2:30 | about 70 s |
| Shlok | 7 Setup · 8 Results · 9 Clipping result · 10 End-sem plan · 11 Thank you | 2:30 – 4:14 | about 70 s |

## How to use this sheet

- Each line has a **start** and an **end** time. Start speaking at the start time and be finished by the end time.
- **Cue** describes what is on screen at that moment. If your editor's timer and the video drift a little, follow the cue, not the clock.
- Speak calmly, at about the pace of a news reader. Every line fits its window at a normal speed.
- Silent gaps are intentional: the animation speaks for itself there.

### Recording

1. Record **each line as its own clip**, in a quiet room, with a phone about 20 cm from your mouth. Name the clips by line number: `P1.m4a`, `P2.m4a` … `A1.m4a` … `S1.m4a` ….
2. Leave about half a second of silence before and after each line, so it can be trimmed cleanly.
3. If you stumble, say the line again in the same clip; keep the best take when editing.

### Editing (CapCut, Clipchamp, DaVinci Resolve or iMovie)

1. Import `inside_explainer.mp4` and put it on the video track.
2. Put each audio clip on an audio track and drag its start to the line's **start time** (zoom the timeline in to place it to within half a second).
3. If a clip runs past its **end time**, trim the pauses inside it, or speed it up by at most 1.1×.
4. Optional: add quiet background music under everything, at about −28 dB.
5. Export as MP4, 1080p, 30 fps.

---

## PRAPTI: 0:00 – 1:00

### Slide 1 · Title (0:00 – 0:14)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| P1 | 0:01 | 0:13 | the big word "INSIDE" rises in | "Hello everyone. We are Prapti Chotaliya, Aayaan Ali Khan and Shlok Bhavsar. For our CS307 mid-semester project, we implemented INSIDE, an ICLR 2024 paper on detecting hallucinations in large language models." |

### Slide 2 · The problem (0:14 – 0:38)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| P2 | 0:14.5 | 0:18 | the heading "LLMs sound confident even when wrong" appears | "Large language models sound equally confident when they are wrong." |
| P3 | 0:18.5 | 0:23.5 | blue boxes "1932" stack up under "Model knows" | "Ask the same question five times. When the model knows, all the answers agree." |
| P4 | 0:24 | 0:27 | gold boxes with different years scatter under "Model is guessing" | "When it is guessing, the answers scatter." |
| P5 | 0:29 | 0:35 | the italic line "Idea: sample K answers…" appears | "So the idea is simple: sample several answers, and measure how much they disagree." |

### Slide 3 · The key idea (0:38 – 1:00)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| P6 | 0:38.5 | 0:43 | the heading appears and the grey layer stack builds up | "INSIDE does not compare the words. It looks inside the model." |
| P7 | 0:43.5 | 0:49 | the middle layer turns gold and a gold line runs to the right | "From the middle layer, it takes the hidden vector of each answer's last token." |
| P8 | 0:49.5 | 0:54 | blue columns fill in one by one | "Ten answers give ten vectors, which form a matrix Z." |
| P9 | 0:55 | 0:58 | the label "Z = [z1 … z10]" is showing | "Aayaan will now explain the math." |

---

## AAYAAN: 1:00 – 2:30

### Slide 4 · The math (1:00 – 1:42)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| A1 | 1:01.5 | 1:04 | blue dots appear in the white box | "Each dot is one answer's vector." |
| A2 | 1:04.5 | 1:10.5 | the square grid fills in cell by cell | "From these we compute a ten-by-ten covariance matrix, which captures how the answers vary together." |
| A3 | 1:11 | 1:18 | gold bars grow, then the formula appears | "Its eigenvalues measure the spread in each direction. EigenScore is simply the average log of these eigenvalues." |
| A4 | 1:19 | 1:25.5 | dots pull together, bars shrink, the slider moves to "low · trust" | "When the answers agree, the eigenvalues collapse and the score is low: we can trust the answer." |
| A5 | 1:26.5 | 1:33 | dots spread out, bars grow, the slider moves to "high · hallucination" | "When they disagree, the eigenvalues grow, the score is high, and the answer is likely a hallucination." |
| A6 | 1:34 | 1:41 | the small note about α appears at the top right | "A small alpha keeps the matrix invertible, and since it is only ten by ten, it is cheap to compute." |

### Slide 5 · Blind spot and feature clipping (1:42 – 2:10)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| A7 | 1:42.5 | 1:49.5 | ten identical gold boxes "1928" appear | "But there is a blind spot. If the model is confidently wrong, all ten answers are the same, and the score trusts them." |
| A8 | 1:50 | 1:56.5 | the heading changes to "Fix: test-time feature clipping"; the histogram builds | "The paper's fix is feature clipping. These are one neuron's activations; the gold bars are rare extreme values." |
| A9 | 1:57 | 2:04 | the gold bars slide in to the dashed lines; the formula appears on the right | "These extremes are clipped to the 0.2 and 99.8 percentile thresholds, estimated from the last three thousand tokens." |
| A10 | 2:04.5 | 2:08.5 | the gold line "→ less overconfidence…" appears | "This reduces overconfidence, so hidden errors start to show." |

### Slide 6 · Their code vs. our code (2:10 – 2:30)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| A11 | 2:10.5 | 2:13 | the column titles "The paper's code" and "Our version" appear | "On the left, the authors' code. On the right, our version." |
| A12 | 2:13.3 | 2:16 | the right card "All 12 fixed" slides in | "Their code crashed; we fixed all twelve bugs." |
| A13 | 2:16.3 | 2:19 | the right card "Implemented from §3.2" slides in | "Feature clipping was missing; we implemented it." |
| A14 | 2:19.3 | 2:22 | the right card "4-bit 7B model on an 8 GB laptop GPU" slides in | "We ran the model in 4-bit on an 8 GB laptop GPU," |
| A15 | 2:22.3 | 2:24.5 | the right card "8 automated tests" slides in | "and added eight automated tests." |
| A16 | 2:25.5 | 2:28.5 | all four rows are showing | "Shlok will now compare our experiment and results." |

---

## SHLOK: 2:30 – 4:14

### Slide 7 · Their setup vs. our setup (2:30 – 2:52)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| S1 | 2:31.8 | 2:35.5 | the "Model" row appears, left then right | "We used Llama-2 in 4-bit instead of LLaMA in 16-bit," |
| S2 | 2:36.3 | 2:40.8 | the right "Data" card with the "smaller" tag slides in | "and 250 questions instead of nearly eight thousand, because of our hardware." |
| S3 | 2:41 | 2:47 | rows with blue "same" tags appear | "Everything else, the sampling, the method settings and the correctness rule, is exactly the same." |

### Slide 8 · Their results vs. our results (2:52 – 3:18)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| S4 | 2:54 | 3:00 | the EigenScore bar grows on the left (80.8), then on the right (83.5) | "Here are the results side by side. EigenScore scores 80.8 in the paper and 83.5 in ours." |
| S5 | 3:00.5 | 3:06 | the remaining bars grow, alternating left and right | "Lexical similarity, entropy, perplexity and energy follow in the same order on both sides." |
| S6 | 3:09.6 | 3:14.5 | the blue banner "Same ranking, reproduced exactly" appears | "So the paper's ranking is reproduced exactly, with EigenScore the best detector." |

### Slide 9 · Does feature clipping help? (3:18 – 3:38)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| S7 | 3:20 | 3:23 | the gold "+1.1" and its diamond appear on the left | "The paper reports that clipping adds 1.1 points." |
| S8 | 3:23.4 | 3:29 | "−1.0" appears on the right and the blue interval bar stretches out | "We measured minus 1.0, but our 95 percent interval runs from minus 4.7 to plus 2.6." |
| S9 | 3:29.5 | 3:35.5 | the line "Our interval contains both 0 and +1.1…" appears | "It covers both zero and the paper's gain, so with 250 questions the result is inconclusive." |

### Slide 10 · End-semester plan (3:38 – 4:04)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| S10 | 3:39.5 | 3:44.5 | one layer in the stack turns gold: "paper: one hand-picked layer" | "By end-semester, we will improve the method. The paper reads only one hand-picked layer." |
| S11 | 3:45 | 3:51 | a band of layers turns gold and their lines merge into one dot | "We will combine a band of middle layers into one score, so no layer needs hand-tuning." |
| S12 | 3:53 | 3:59 | the cards "+ Hybrid detector" and "+ Adaptive K" slide in | "We will also add token confidence to catch confident mistakes, and adaptive sampling to reduce cost." |

### Slide 11 · Thank you (4:04 – 4:14)

| # | Start | End | Cue on screen | Say |
|---|---|---|---|---|
| S13 | 4:05 | 4:07 | "Thank you" rises in | "Thank you for watching." |

---

### If you want more even speaking time

Prapti speaks the least, about 45 s. To even it out, move **Slide 4 (A1–A6)** to Prapti. The split then becomes Prapti about 75 s, Aayaan about 45 s and Shlok about 70 s. In that case Prapti says P9 as "Now, the math.", and the handover to Aayaan moves to the start of A7.

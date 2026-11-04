# Design-project video: voiceover scripts (ElevenLabs v2)

Video: `design_video_silent.mp4`, 2:00, 1920×1080. Order: **Prapti → Aayaan → Shlok**.

Each person generates **one audio file** from their block below and sends it back. The `<break>` tags already hold the right gaps, so each file plays in sync if it starts at that person’s first cue. When merging, I place every line on its cue and nudge a line by a few percent only if it runs long.

## ElevenLabs settings (same for all three)

| Setting | Value |
|---|---|
| Model | **Eleven Multilingual v2** |
| Voice | your own cloned voice |
| Speed | **1.0** (if your file is longer than the target below, re-generate at 1.05–1.1) |
| Stability | 50% |
| Similarity | 80% |
| Style exaggeration | 0% |
| Speaker boost | on |
| Output | MP3 44.1 kHz 192 kbps (or WAV) |

Paste the block exactly as written. Numbers are spelled out on purpose so they are pronounced correctly. Don’t add or remove `<break>` tags.

---

## Prapti: 0:00.7 → 0:37.4  (target file length ≈ 37 s)

```
Hello everyone. I'm Prapti Chotaliya, and my teammates are Aayaan Ali Khan and Shlok Bhavsar. <break time="1.4s" />
Large language models often give wrong answers, in the same confident tone as correct ones. <break time="0.6s" />
We want to flag likely wrong answers, without an answer key, on an eight gigabyte GPU. <break time="1.5s" />
Our idea is simple. We ask the model the same question ten times. <break time="0.6s" />
If it knows, the answers agree. If it is guessing, they scatter. <break time="0.6s" />
We compare them through the model's internal states, not their words.
```

| # | Scene | Starts at | Must end by | Line | Tone |
|---|---|---|---|---|---|
| P1 | Title | 0:00.7 | 0:08.7 | Hello everyone. I'm Prapti Chotaliya, and my teammates are Aayaan Ali Khan and Shlok Bhavsar. | warm, welcoming; small pause after “Hello everyone.” |
| P2 | 1 · Problem | 0:08.9 | 0:14.9 | Large language models often give wrong answers, in the same confident tone as correct ones. | serious, setting up the problem |
| P3 | 1 · Problem | 0:15.1 | 0:22.7 | We want to flag likely wrong answers, without an answer key, on an eight gigabyte GPU. | clear and purposeful; slight stress on “without an answer key” |
| P4 | 2 · The idea | 0:22.9 | 0:28.0 | Our idea is simple. We ask the model the same question ten times. | lighter, inviting |
| P5 | 2 · The idea | 0:28.2 | 0:33.2 | If it knows, the answers agree. If it is guessing, they scatter. | contrast: steady on “agree”, a little lift on “scatter” |
| P6 | 2 · The idea | 0:33.4 | 0:38.7 | We compare them through the model's internal states, not their words. | firm; stress “internal states” |

---

## Aayaan: 0:38.9 → 1:20.8  (target file length ≈ 42 s)

```
Each answer becomes a vector inside the network, and we measure how spread out these vectors are. <break time="0.6s" />
A tight cluster gives small eigenvalues and a low score. <break time="0.6s" />
A wide spread gives large eigenvalues and a high score, so the answer is flagged. <break time="1.5s" />
We use Llama 2, seven billion, and read layer sixteen of thirty two, at the last token of each answer. <break time="0.6s" />
Ten answers give a ten by four thousand ninety six matrix, which reduces to ten eigenvalues, and one score. <break time="1.5s" />
To fit an eight gigabyte GPU, we load the model in four bit, from thirteen point five down to three point nine gigabytes.
```

| # | Scene | Starts at | Must end by | Line | Tone |
|---|---|---|---|---|---|
| A1 | 3 · Eigenvalues | 0:38.9 | 0:45.4 | Each answer becomes a vector inside the network, and we measure how spread out these vectors are. | explanatory, even pace |
| A2 | 3 · Eigenvalues | 0:45.6 | 0:49.6 | A tight cluster gives small eigenvalues and a low score. | calm, matter-of-fact |
| A3 | 3 · Eigenvalues | 0:49.8 | 0:56.2 | A wide spread gives large eigenvalues and a high score, so the answer is flagged. | slightly stronger; land on “flagged” |
| A4 | 4 · Pipeline | 0:56.5 | 1:03.9 | We use Llama 2, seven billion, and read layer sixteen of thirty two, at the last token of each answer. | technical, confident; clear on “sixteen of thirty two” |
| A5 | 4 · Pipeline | 1:04.1 | 1:12.6 | Ten answers give a ten by four thousand ninety six matrix, which reduces to ten eigenvalues, and one score. | steady listing pace, do not rush the numbers |
| A6 | 5 · 8 GB GPU | 1:12.8 | 1:21.2 | To fit an eight gigabyte GPU, we load the model in four bit, from thirteen point five down to three point nine gigabytes. | practical; small lift on “three point nine” |

---

## Shlok: 1:21.4 → 1:57.0  (target file length ≈ 36 s)

```
A hook keeps only the vector we need, answers are generated two at a time, and eight unit tests check the maths. <break time="1.5s" />
On two hundred and fifty test questions, the model was wrong on one hundred and fifty eight. <break time="0.6s" />
Our detection score, AUROC, is eighty three point five percent, where random guessing gives fifty. <break time="1.5s" />
As future directions, we will combine a band of middle layers instead of one, <break time="0.6s" />
add token confidence, and sample fewer answers when they clearly agree. <break time="1.5s" />
Thank you.
```

| # | Scene | Starts at | Must end by | Line | Tone |
|---|---|---|---|---|---|
| S1 | 5 · 8 GB GPU | 1:21.4 | 1:29.9 | A hook keeps only the vector we need, answers are generated two at a time, and eight unit tests check the maths. | quick list of three things |
| S2 | 6 · Results | 1:30.1 | 1:36.2 | On two hundred and fifty test questions, the model was wrong on one hundred and fifty eight. | factual |
| S3 | 6 · Results | 1:36.4 | 1:44.2 | Our detection score, AUROC, is eighty three point five percent, where random guessing gives fifty. | confident; a small lift on “eighty three point five” |
| S4 | 7 · Future directions | 1:44.4 | 1:50.0 | As future directions, we will combine a band of middle layers instead of one, | forward-looking |
| S5 | 7 · Future directions | 1:50.2 | 1:56.2 | add token confidence, and sample fewer answers when they clearly agree. | wrap-up tone, falling at the end |
| S6 | Thank you | 1:56.4 | 1:59.5 | Thank you. | warm, closing |

---

## Whole timeline at a glance

| Time | Scene | Speaker |
|---|---|---|
| 0:00.0–0:08.2 | Title | Prapti |
| 0:08.2–0:22.2 | 1 · Problem | Prapti |
| 0:22.2–0:38.2 | 2 · The idea | Prapti |
| 0:38.2–0:55.8 | 3 · Eigenvalues | Aayaan |
| 0:55.8–1:12.1 | 4 · Pipeline | Aayaan |
| 1:12.1–1:29.4 | 5 · 8 GB GPU | Aayaan → Shlok |
| 1:29.4–1:43.7 | 6 · Results | Shlok |
| 1:43.7–1:55.7 | 7 · Future directions | Shlok |
| 1:55.7–2:00.0 | Thank you | Shlok |

Tip: listen once before sending. If a word is mispronounced (for example “Aayaan” or “Chotaliya”), re-generate just that take; spelling it phonetically (“Aa-yaan”, “Cho-ta-li-ya”) usually fixes it.

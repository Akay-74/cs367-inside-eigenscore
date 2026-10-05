# Shlok's voiceover with ElevenLabs

Your part covers 2:30 to 4:14 of the video: 13 lines, about 75 seconds of speech.
You generate **one audio file** in ElevenLabs and send it to me. I split it into the 13 lines and place each line at its exact moment in the video. If a line runs a little long, I speed it up slightly without changing the pitch. You don't need to time anything yourself.

---

## 1 · ElevenLabs settings

| Setting | Value | Why |
|---|---|---|
| Voice | your cloned voice | |
| Model | **Eleven v4** | follows audio tags like `[long pause]` reliably |
| Speed | **1.0** (use 1.05 if a re-take is needed) | |
| Stability | **Natural** (or Robust if lines come out over-acted) | steady, formal academic delivery |
| Similarity | **80–85%** | sounds like you |
| Speaker boost | **on** | |
| Output | MP3 44.1 kHz 192 kbps, or WAV | |

v4 doesn't use `<break>` tags. It uses audio tags in square brackets instead: `[long pause]` separates the lines, and `[calm, confident]` at the start sets your tone. The long pauses between lines are how I find where each line starts, so keep them.

Some settings in the table (Stability, Similarity, Speaker boost) may not appear for v4. If one isn't there, skip it.

---

## 2 · Paste this text, exactly as it is, into ElevenLabs

Numbers are written out in words and "EigenScore" is spaced as "Eigen Score", so the voice pronounces them correctly. The `[long pause]` tags create the gaps I use to cut the lines apart. Don't remove them.

```
[calm, confident] We used Llama 2 in four-bit, instead of Llama in sixteen-bit, [long pause]
and two hundred and fifty questions instead of nearly eight thousand, because of our hardware. [long pause]
Everything else, the sampling, the method settings, and the correctness rule, is exactly the same. [long pause]
Here are the results, side by side. Eigen Score scores eighty point eight in the paper, and eighty-three point five in ours. [long pause]
Lexical similarity, entropy, perplexity, and energy follow in the same order on both sides. [long pause]
So the paper's ranking is reproduced exactly, with Eigen Score the best detector. [long pause]
The paper reports that clipping adds one point one points. [long pause]
We measured minus one point zero, but our ninety-five percent interval runs from minus four point seven, to plus two point six. [long pause]
It covers both zero and the paper's gain, so with two hundred and fifty questions, the result is inconclusive. [long pause]
By end-semester, we will improve the method. The paper reads only one hand-picked layer. [long pause]
We will combine a band of middle layers into one score, so no layer needs hand-tuning. [long pause]
We will also add token confidence to catch confident mistakes, and adaptive sampling to reduce cost. [long pause]
Thank you for watching.
```

Generate it, listen once, and download it.

---

## 3 · What each line does, and when it plays

You don't need to match these times. They show where each line lands and the tone it should have, so you can judge whether a take sounds right.

| # | Plays at | Must end by | Line | Tone |
|---|---|---|---|---|
| S1 | 2:31.8 | 2:35.9 | We used Llama 2 in four-bit, instead of Llama in sixteen-bit, | matter-of-fact; slight rise at the end because the sentence continues |
| S2 | 2:36.2 | 2:41.6 | and two hundred and fifty questions instead of nearly eight thousand, because of our hardware. | explanatory, a little lower on "because of our hardware" |
| S3 | 2:41.9 | 2:51.0 | Everything else, the sampling, the method settings, and the correctness rule, is exactly the same. | reassuring; light stress on "exactly the same" |
| S4 | 2:53.0 | 3:00.7 | Here are the results, side by side. Eigen Score scores eighty point eight in the paper, and eighty-three point five in ours. | confident; a small lift on "eighty-three point five" |
| S5 | 3:01.0 | 3:09.2 | Lexical similarity, entropy, perplexity, and energy follow in the same order on both sides. | even listing pace |
| S6 | 3:09.6 | 3:17.0 | So the paper's ranking is reproduced exactly, with Eigen Score the best detector. | the key result: clear and firm |
| S7 | 3:19.6 | 3:22.9 | The paper reports that clipping adds one point one points. | neutral, sets up a contrast |
| S8 | 3:23.2 | 3:31.0 | We measured minus one point zero, but our ninety-five percent interval runs from minus four point seven, to plus two point six. | careful, slightly slower on the numbers |
| S9 | 3:31.3 | 3:37.6 | It covers both zero and the paper's gain, so with two hundred and fifty questions, the result is inconclusive. | honest and calm, not apologetic |
| S10 | 3:39.5 | 3:44.9 | By end-semester, we will improve the method. The paper reads only one hand-picked layer. | forward-looking, a fresh start |
| S11 | 3:45.2 | 3:52.2 | We will combine a band of middle layers into one score, so no layer needs hand-tuning. | the main idea, clearly stated |
| S12 | 3:52.5 | 4:03.5 | We will also add token confidence to catch confident mistakes, and adaptive sampling to reduce cost. | steady and wrapping up |
| S13 | 4:05.0 | 4:13.5 | Thank you for watching. | warm and short |

The slots are in `voice_slots.json`. Compared with the subtitle guide, some lines start slightly earlier or run into the following pause. This is because spelled-out numbers take longer to say than they take to read.

---

## 4 · If a line sounds wrong

- **Wrong word stress or mispronunciation:** regenerate just that one line, by itself and without the `[long pause]` tag. Name the file by its number, for example `S8.mp3`, and send it along. I'll replace that line.
- **"Llama" sounds odd:** try writing it as "Lama".
- **Everything feels rushed or too slow:** regenerate the whole text at Speed 0.95 or 1.05.
- **Too dramatic or uneven:** remove `[calm, confident]`, or switch Stability to Robust.
- **v4 adds a long pause in the middle of a line:** that's fine, I always cut at the 12 longest pauses. If a line still gets cut in the wrong place, generate the lines as separate files instead (see the next point).
- **Prefer separate files?** You can generate each line on its own and name them `S1.mp3` … `S13.mp3`. That works too.

---

## 5 · Send me

- the downloaded file, for example `shlok_full.mp3`
- any separately regenerated lines (`S8.mp3` and so on)

I'll run:

```
python3 place_voice.py S shlok_full.mp3     # splits, times and loudness-matches your part
python3 place_voice.py --mux                # adds it to the video
```

You'll get back `inside_with_voice.mp4` with your part in place, plus a report showing where each line landed and whether any needed speeding up. Prapti's and Aayaan's recordings can go through the same tool later (`P` and `A`), so all three voices end up perfectly in sync.

# Mid-semester presentation video: plan

Course: **CS307 (Artificial Intelligence)**
Group: Aayaan Ali Khan (202411001), Shlok Bhavsar (202411018), Prapti Chotaliya (202411030)
Paper: Chen et al., *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection*, ICLR 2024

Slides (draft): https://claude.ai/artifact/JY8TehiMRB8dqFrsbiWvKP
The full voiceover script is in each slide's speaker notes. A copy of the slide sources is in `presentation/deck/`.

## 1. Format

**Narrated slides with restrained motion.** This is not a fully animated video.

- 16:9, 1920×1080, 30 fps, MP4 (H.264, AAC audio).
- Each slide is on screen while its speaker narrates it. Elements fade in one at a time to match the narration (fade only: no flying, bouncing or zooming).
- Two short animated segments carry the method explanation. Both are optional, and the static slide works without them:
  1. **EigenScore (slide 7):** ten points move from a tight cluster to a spread cloud while the eigenvalue bars below them grow. About 6 seconds.
  2. **Feature clipping (slide 10):** the histogram's gold tails slide inward and stop at the dashed thresholds. About 4 seconds.
  These can be made in PowerPoint/Keynote (Morph or Magic Move), or with Manim if one of us wants a cleaner result.
- An optional 10-second screen recording of the tests passing and the generation command running, placed after slide 13.

Visual style: navy, ivory and muted gold; Source Serif 4 for headings, IBM Plex Sans for body text, IBM Plex Mono for formulas. No stock photos, sound effects or background music, or, if music is used, quiet and only under the title and closing slides.

## 2. Running order, speakers and timing

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Title and group members | Aayaan | 0:35 |
| 2 | Outline | Aayaan | 0:20 |
| 3 | Hallucination: fluent, confident and wrong | Aayaan | 0:35 |
| 4 | Where existing detectors look | Aayaan | 0:40 |
| 5 | INSIDE at a glance | Aayaan | 0:30 |
| 6 | From one question to one score (pipeline) | Shlok | 0:45 |
| 7 | EigenScore: spread measured by eigenvalues | Shlok | 0:45 |
| 8 | Why a log-determinant | Shlok | 0:35 |
| 9 | The blind spot: self-consistent hallucinations | Shlok | 0:25 |
| 10 | Test-time feature clipping | Shlok | 0:45 |
| 11 | How detectors are compared | Shlok | 0:35 |
| 12 | The official code did not run | Prapti | 0:40 |
| 13 | What we built | Prapti | 0:50 |
| 14 | Experimental setup: paper and ours | Prapti | 0:35 |
| 15 | The paper's ranking reproduces exactly | Prapti | 0:45 |
| 16 | EigenScore against the strongest baseline | Prapti | 0:40 |
| 17 | Feature clipping: inconclusive at this scale | Prapti | 0:45 |
| 18 | Limitations | Shlok | 0:40 |
| 19 | End-semester extension | Aayaan | 0:55 |
| 20 | Plan and success criteria | Aayaan | 0:30 |
| 21 | Thank you and references | Aayaan | 0:15 |
| | **Total** | | **about 13:00** |

Speaking time is balanced: Aayaan about 4:20, Shlok about 4:30, Prapti about 4:15.

**If there is a time limit of 10 minutes**, remove slides 8, 11 and 14 (about 1:45) and shorten the notes on slides 13 and 19.

Names are spoken once, on the title slide. Roll numbers are shown on screen only.

## 3. Production steps

1. **Finalise the slides.** Fill in the placeholders (the list is below) and confirm the end-semester direction.
2. **Rehearse.** Each speaker reads their notes aloud once against a timer and rewords anything that does not sound natural for them. Keep the facts and numbers unchanged.
3. **Record the audio** separately for each speaker: a quiet room, a phone or headset microphone 15 to 20 cm from the mouth, one WAV or M4A file per slide, named `06_shlok.wav` and so on. Recording each slide separately makes retakes easy.
4. **Export the slides** as 1920×1080 images or PDF (the deck has Download/Export), or present them full screen and screen-record with OBS.
5. **Assemble** the video in a simple editor (DaVinci Resolve, Clipchamp, iMovie, or PowerPoint's *Record* feature). Place each slide for the length of its audio, add 0.5 s crossfades between slides, and insert the two animated segments.
6. **Finish:** normalise the audio to about −16 LUFS, add captions (auto-generated, then corrected), and export as MP4.
7. **Check** the whole video once on a phone and once on a laptop before submitting.

## 4. Placeholders to fill

- Slide 3: one real CoQA question, its reference answer, and three sampled answers that disagree. These come from `data/output/llama2-7b-hf_coqa_10/0.pkl`.
- Slide 20: actual week dates for the end-semester timeline.
- Opening greeting ("Good morning") if the course expects a different one.

## 5. Decisions for the group

- **End-semester direction.** The draft proposes a multi-layer, normalised EigenScore as the main contribution, a hybrid EigenScore + LN-Entropy detector as the second, and adaptive K as a stretch goal. Choose one main idea before recording, because the video commits us to it.
- **Course code.** The repository and report say CS367, while the course is CS307. The slides use CS307. Fix the report before final submission.

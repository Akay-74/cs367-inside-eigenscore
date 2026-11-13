# CS367 Endsem: what to submit and where

Team: Aayaan Ali Khan (202411001), Shlok Bhavsar (202411018), Prapti Chotaliya (202411030)
Guide: Dr. Venkata Phanikrishna

Only the team coordinator submits.

## Files in this folder

| File | Use |
|---|---|
| `Agreement_Meets_Confidence_paper.pdf` | The 5-page manuscript (IEEE SPL template). Upload this to Turnitin. |
| `overleaf_upload.zip` | Overleaf project: main.tex, IEEEtran.cls, two figures. |
| `Agreement_Meets_Confidence_video.mp4` | Video, 10 slides (3 existing work, 7 extension), about 6.5 min. |
| `Agreement_Meets_Confidence.pptx` | The same 10 slides with speaker notes. |
| `demo_gui/app.html` | Interactive demo comparing EigenScore with our method. Open it in a browser; it works offline. |
| `source_code.zip` | All extension code, frozen plans, results (without the 7.8 GB embedding caches). |

## Steps

1. **Overleaf.** Open the shared template project, replace its files with the contents of
   `overleaf_upload.zip` (main.tex, IEEEtran.cls, the `figures/` folder) and click Recompile.
   You should get 5 pages. Then use Share → "Anyone with this link can view" and copy the link.
2. **Form.** Paste the Overleaf link, upload or link the video, and fill in the self-grades.
3. **Turnitin.** Upload `Agreement_Meets_Confidence_paper.pdf` and download the similarity report
   if the form asks for it.
4. **Self-grading.** Use whole numbers only, and give each of the three members a different grade.
5. **Presentation to ma'am.** Open `demo_gui/app.html`. The "Results" tab shows the ROC
   curves and the alarm-budget slider. The "Question explorer" tab shows individual questions.

## Headline numbers (in case the form asks)

- TRI beats EigenScore by +2.7 AUROC on both held-out datasets (TriviaQA, NQ-Open) and by +4.5 on CoQA.
  All three gains are significant (paired bootstrap, 10,000 resamples, Holm-corrected).
- Mean AUROC: TRI 82.4 vs EigenScore 79.1.
- On NQ-Open, at a 5% false-alarm rate, TRI catches 39.7% of hallucinations; EigenScore catches 28.7%.

# Video narration (10 slides, about 8 minutes)

There is one paragraph per slide. Each slide stays on screen exactly as long as its audio.
Numbers are written as words so the voice reads them correctly, and "EigenScore" is written
"Eigen Score".

Suggested split: Aayaan reads slides 1–3, Shlok reads 4–7, Prapti reads 8–10.

---

**Slide 1 · Problem statement**
Large language models often give answers that are fluent, confident, and wrong. In our example, the model states that a bridge opened in nineteen twenty-eight, with complete confidence, even though that is not true. A reader cannot tell such an answer apart from a correct one. And when the model is deployed, there is no answer key to check against. So the problem we address is this: give every answer a hallucination score, using only the model itself, so that unreliable answers can be flagged before they reach the user.

**Slide 2 · Existing detectors**
Existing detectors look in one of three places. The first group uses token probabilities, such as perplexity and length-normalised entropy, which measure how unsure the model is about each word. The second group samples several answers and compares their words, for example with lexical similarity. The third group, which includes the paper we build on, INSIDE from ICLR twenty twenty-four, compares the sampled answers inside the network, where their meaning is still densely encoded.

**Slide 3 · EigenScore and what it cannot see**
INSIDE's Eigen Score samples ten answers to the same question, reads each answer's hidden state at the middle layer, and measures the volume of that cloud of ten points using the eigenvalues of their covariance. At midsem we repaired the authors' code and reproduced the method on an eight gigabyte laptop GPU, where Eigen Score reached eighty-three point five AUROC, the best of five methods. But it relies on a single kind of evidence. If the model repeats the same wrong answer ten times, the spread is zero and Eigen Score trusts it. And it ignores the confidence the model reports for every word it writes.

**Slide 4 · Our idea**
Our idea starts from a simple observation. The same ten samples already contain three different kinds of evidence. Internal evidence: how far the answers spread in the hidden states, which is Eigen Score. Surface evidence: how much the ten answers overlap in their words. And token evidence: how confident the model was about each word. If these signals fail on different questions, combining them should beat any one of them, at no extra cost: no new generations and no training.

**Slide 5 · The TRI method**
Our method, TRI, is a rank fusion. For every question we rank it by each of the three signals and average the three ranks. We use ranks because the signals live on completely different scales, and ranks make them comparable. The weights are equal, and nothing is fitted to the data, so the method needs no labels at all. We also propose a companion method, consistency then confidence. It ranks by text agreement, and only where two questions tie, often because all ten answers are identical, it uses token confidence to break the tie. Because ties receive only half credit in AUROC, breaking them with an informative signal can only help.

**Slide 6 · Why it works**
Why does this work? Because the signals make different mistakes. Their rank correlations range from only zero point six two to zero point nine. Token confidence is the most independent, and it sees exactly what Eigen Score misses. Among questions where all ten answers agree, token confidence still separates right from wrong answers, with a within-tie AUC between zero point six five and zero point seven three, while Eigen Score manages only zero point five nine on CoQA. Our ablation shows that on the hardest dataset, all three signals contribute.

**Slide 7 · Experimental protocol**
To make sure our improvement is real, we followed a strict protocol. We ran Llama two, seven billion parameters, in four-bit precision on an eight gigabyte laptop GPU, with ten samples for each of one thousand questions per dataset. We developed every idea on CoQA only. Before looking at any other result, we froze our final methods in git. Only then did we test them on two unseen datasets, TriviaQA and Natural Questions, using a paired bootstrap with ten thousand resamples and a correction for multiple comparisons.

**Slide 8 · Results**
Here are the results. Eigen Score averages seventy-nine point one AUROC across the three datasets, and text agreement averages eighty-one point five. Our fusion, TRI, is the strongest on average at eighty-two point four, and it never loses to Eigen Score. The biggest test is Natural Questions, where the model answers only twenty-seven percent of questions correctly. There, every single signal stays below seventy-seven, while TRI reaches seventy-nine point five.

**Slide 9 · Held-out confirmation**
This plot shows the gain over Eigen Score with ninety-five percent confidence intervals. On the two datasets that were never used to design the method, TRI improves on Eigen Score by two point seven points each time, and both gains remain significant after correction. Text agreement alone is not reliable: it fails completely on Natural Questions. And our companion method beats text agreement on both unseen datasets. TRI is the only method that beats Eigen Score on every dataset.

**Slide 10 · Conclusion**
To conclude: a simple, label-free fusion of internal, surface and token signals beats Eigen Score by two point seven to four point five AUROC on three datasets, and the gains held on data frozen out of development. We also found that Eigen Score's choice of layer barely matters, and that averaging layers, normalising embeddings or tracking cross-layer structure did not help. Our study uses one seven billion parameter model and ROUGE-based labels, so the next steps are a second model and labels from an LLM judge. Thank you.

# Video narration (10 slides, target ≈ 9 minutes)

There is one paragraph per slide. Each slide stays on screen exactly as long as its audio
clip (plus a short pause), so the timing follows the voice.

Speakers will be assigned after the results are in. When generating with ElevenLabs:
- Use one take per speaker, with `[long pause]` between that speaker's slides.
- Numbers are written as words so the voice says them correctly.
- "EigenScore" is written "Eigen Score".

---

**Slide 1 · Problem statement** (≈ 45 s)
Large language models often give answers that are fluent, confident, and wrong. In our example, the model states that a bridge opened in nineteen twenty-eight, with complete confidence, even though that is not true. A user cannot tell such an answer apart from a correct one by reading it. And when the model is deployed, there is no answer key to check against. So the problem we address is this: give every answer a hallucination score, using only the model itself, so that unreliable answers can be flagged before they reach the user.

**Slide 2 · Existing detectors** (≈ 45 s)
Existing detectors look in one of three places. The first group uses token probabilities, such as perplexity and length-normalised entropy, which measure how unsure the model is about each word. But a model can be confidently wrong. The second group samples several answers and compares their words, for example with lexical similarity. But paraphrases then look like disagreement. The third group, which includes the paper we build on, INSIDE from ICLR twenty twenty-four, compares the sampled answers inside the network, where their meaning is still densely encoded.

**Slide 3 · EigenScore and its blind spots** (≈ 60 s)
INSIDE's Eigen Score samples ten answers to the same question. For each answer, it reads the hidden state of the last token at the middle layer, layer sixteen of thirty-two. It then measures the volume of that cloud of ten points, using the eigenvalues of their covariance matrix. If the answers agree, the volume collapses and the score is low. If they scatter, the score is high and the answer is flagged. At midsem we repaired the authors' code and reproduced the method on an eight gigabyte laptop GPU: Eigen Score reached eighty-three point five AUROC, the best of five methods. But it has two blind spots. It bets on one hand-picked layer, even though detection quality changes with depth. And it measures only how far the answers spread, not how they are arranged.

**Slide 4 · Extension overview** (≈ 50 s)
Our extension tests four training-free ideas. The first, M1, averages the score over a band of layers. The second, M2, equalises the embedding dimensions before the covariance is formed. The third, M3, fuses the score with word-level confidence. The fourth, cross-depth stability, is a new concept that asks whether the arrangement of the answers survives from one layer to the next. To make the comparison trustworthy, every score is computed from the same ten samples per question, so all comparisons are paired. The layer band and the statistical tests were committed to git before any result existed, and there is a single primary test: M1 against the original Eigen Score.

**Slide 5 · M1, Multi-layer EigenScore** (≈ 50 s)
M1 is the multi-layer Eigen Score. Instead of reading only layer sixteen, it averages the score over the middle half of the network, layers eight to twenty-four. The band is fixed by a simple rule, a quarter to three quarters of the depth, and it was chosen before we saw any result. The reasoning is that every layer gives a noisy reading of the same underlying uncertainty, and averaging reduces that noise. It also avoids a subtle trap: if you pick the best layer after seeing the results, its performance looks better than it really is. A fixed band avoids that bias.

**Slide 6 · M2 and M3** (≈ 50 s)
M2 addresses a known property of language models: a few fixed dimensions in the hidden states carry values orders of magnitude larger than the rest, and these can dominate the covariance. M2a scales each answer's embedding to unit length. M2b standardises every dimension, using statistics from previous questions only, so that no information leaks from the current one. M3 is a hybrid. Eigen Score reads the spread of internal states, while length-normalised entropy reads word-level confidence. Because they rely on different signals, they can make different mistakes, so we combine their ranks with equal weights, with no tuning and no labels.

**Slide 7 · XDS, the new concept** (≈ 60 s)
Our new idea is cross-depth stability. At each layer, we record which answers resemble which, through the forty-five pairwise distances between the ten answers. We then measure how strongly that pattern carries over to the next layer, a technique called representational similarity analysis. Our hypothesis is this. When the model knows the answer, its samples are paraphrases of one fact, so their arrangement stays stable through the layers. When it is making something up, nothing anchors the samples, so their arrangement reshuffles from layer to layer. As the picture shows, two clouds can have exactly the same spread but a different arrangement. Eigen Score cannot see that difference. Cross-depth stability can.

**Slide 8 · Experimental protocol** (≈ 45 s)
We ran Llama two, seven billion parameters, in four-bit precision on an eight gigabyte RTX forty-sixty laptop GPU. We used one thousand questions each from three datasets: CoQA, TriviaQA and Natural Questions. For every question we drew ten samples with the paper's settings. An answer counts as a hallucination when its ROUGE-L overlap with the reference answer is at most zero point five. We report AUROC for every method, and we test each improvement with a paired bootstrap of ten thousand resamples. To fit within eight gigabytes, we recompute the hidden states on the prompt's cached keys and values, and we verified that this matches a full forward pass.

**Slide 9 · Results** (≈ 60 s)
*Written after the runs finish.*

**Slide 10 · Findings and conclusion** (≈ 60 s)
*Written after the runs finish.*

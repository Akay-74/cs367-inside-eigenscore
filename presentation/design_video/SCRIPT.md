# Cue sheet (generated from lines.py)

| ID | Start | End | Scene | Line |
|---|---|---|---|---|
| P1 | 0:00.70 | 0:07.42 | s1 | "Hello everyone. I'm Prapti Chotaliya, and my teammates are Aayaan Ali Khan and Shlok Bhavsar." |
| P2 | 0:08.87 | 0:14.54 | s2 | "Large language models often give wrong answers, in the same confident tone as correct ones." |
| P3 | 0:15.14 | 0:21.49 | s2 | "We want to flag likely wrong answers, without an answer key, on an eight gigabyte GPU." |
| P4 | 0:22.94 | 0:27.58 | s3 | "Our idea is simple. We ask the model the same question ten times." |
| P5 | 0:28.18 | 0:32.79 | s3 | "If it knows, the answers agree. If it is guessing, they scatter." |
| P6 | 0:33.39 | 0:37.43 | s3 | "We compare them through the model's internal states, not their words." |
| A1 | 0:38.88 | 0:45.01 | s4 | "Each answer becomes a vector inside the network, and we measure how spread out these vectors are." |
| A2 | 0:45.61 | 0:49.20 | s4 | "A tight cluster gives small eigenvalues and a low score." |
| A3 | 0:49.80 | 0:55.00 | s4 | "A wide spread gives large eigenvalues and a high score, so the answer is flagged." |
| A4 | 0:56.45 | 1:03.49 | s5 | "We use Llama 2, seven billion, and read layer sixteen of thirty two, at the last token of each answer." |
| A5 | 1:04.09 | 1:11.38 | s5 | "Ten answers give a ten by four thousand ninety six matrix, which reduces to ten eigenvalues, and one score." |
| A6 | 1:12.83 | 1:20.81 | s6 | "To fit an eight gigabyte GPU, we load the model in four bit, from thirteen point five down to three point nine gigabytes." |
| S1 | 1:21.41 | 1:28.69 | s6 | "A hook keeps only the vector we need, answers are generated two at a time, and eight unit tests check the maths." |
| S2 | 1:30.14 | 1:35.81 | s7 | "On two hundred and fifty test questions, the model was wrong on one hundred and fifty eight." |
| S3 | 1:36.41 | 1:42.99 | s7 | "Our detection score, AUROC, is eighty three point five percent, where random guessing gives fifty." |
| S4 | 1:44.44 | 1:49.63 | s8 | "As future directions, we will combine a band of middle layers instead of one," |
| S5 | 1:50.23 | 1:54.97 | s8 | "add token confidence, and sample fewer answers when they clearly agree." |
| S6 | 1:56.42 | 1:56.98 | s9 | "Thank you." |

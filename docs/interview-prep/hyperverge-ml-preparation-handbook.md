# HyperVerge DL/ML Research Intern Preparation Handbook

**Interview:** DL/ML Research Intern (LLMs & VLMs)  
**Focus:** Python, deep learning, PyTorch, Transformers, computer vision, multimodal/document AI, model evaluation, LoRA, alignment, and Vision Language Models.

## How to use this handbook

Answer aloud in your own words. Do not claim that CommentPulse currently uses deep learning: it is an end-to-end **classical ML/MLOps** project using TF-IDF plus LightGBM. Explain the baseline honestly, then show how you would improve it with carefully evaluated deep learning.

## 1. Introduction and CommentPulse

**Interviewer: Tell me about yourself.**  
**Candidate:** I am interested in applied deep learning, language models, computer vision, and production ML. My project, CommentPulse, is a YouTube comment sentiment platform with a Chrome extension, Flask API, TF-IDF plus LightGBM inference, async analytics, DVC, MLflow, and observability. It taught me that useful ML requires reliable data, evaluation, reproducibility, serving, and monitoring in addition to a model.

**Interviewer: Explain CommentPulse in one minute.**  
**Candidate:** The extension collects visible YouTube-style comments and sends them to a Flask API. The API validates input, applies an already fitted TF-IDF vectorizer, and uses LightGBM to predict positive, neutral, or negative sentiment. It also provides charts, topic sentiment, and summaries. Slow analytics run as background jobs through Redis-backed workers, with a local fallback for development. DVC manages reproducible pipeline stages and MLflow records experiments.

**Interviewer: Why TF-IDF plus LightGBM rather than a Transformer?**  
**Candidate:** It is a deliberate baseline: fast, local, low-cost, simple to deploy, and suitable for a limited labeled dataset. Its trade-off is weak contextual understanding, especially for sarcasm, multilingual comments, emojis, and long-range meaning. I would improve data quality first, then benchmark a compact fine-tuned Transformer against it on macro F1, per-class recall, latency, memory, and cost.

**Interviewer: What is TF-IDF?**  
**Candidate:** It weights a term by how frequent it is in one document and how rare it is across the corpus: \(TF\text{-}IDF(t,d)=TF(t,d)\log(N/DF(t))\). It downweights generic terms and highlights discriminative words. N-grams help it capture phrases such as “not good.”

**Interviewer: What is the project’s main weakness?**  
**Candidate:** Dataset size and representativeness limit quality. I would collect more human-labeled, representative comments; define annotation guidelines; deduplicate; use video-level splits to reduce leakage; use active learning; and compare the baseline with a fine-tuned Transformer.

**Interviewer: What is data leakage in this project?**  
**Candidate:** It is training with information unavailable at real inference. Duplicated or near-duplicated comments, or comments from one video appearing in both train and validation sets, can inflate metrics. I would deduplicate and use group-aware splits by source video.

**Interviewer: Why macro F1?**  
**Candidate:** Accuracy can hide poor minority-class performance. Macro F1 computes F1 per class and averages equally, making positive, neutral, and negative quality all visible.

## 2. Python and data foundations

**Interviewer: List, tuple, set, and dictionary?**  
**Candidate:** Lists are ordered and mutable. Tuples are ordered and immutable. Sets contain unique items and give fast membership checks. Dictionaries map keys to values.

**Interviewer: What are generators?**  
**Candidate:** They yield values lazily, so they are memory-efficient for streaming or large datasets.

**Interviewer: Explain `*args` and `**kwargs`.**  
**Candidate:** `*args` captures extra positional arguments as a tuple; `**kwargs` captures keyword arguments as a dictionary.

**Interviewer: Why avoid mutable default arguments?**  
**Candidate:** They are created once and shared across calls.

```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

**Interviewer: Threading versus multiprocessing?**  
**Candidate:** Threads are effective for I/O-bound tasks and share memory. CPU-bound Python tasks often benefit from multiprocessing because CPython’s GIL limits parallel Python-bytecode execution in threads.

**Interviewer: How do you process data too large for memory?**  
**Candidate:** Stream or read in chunks, use generators, avoid copies, batch model inputs, and use a PyTorch Dataset/DataLoader for lazy loading.

**Interviewer: What is broadcasting in NumPy?**  
**Candidate:** Broadcasting allows compatible array shapes to operate together without explicitly copying the smaller array.

**Interviewer: How do you ensure reproducibility?**  
**Candidate:** Seed Python, NumPy, and PyTorch; version code and data; log hyperparameters, metrics, artifacts, and environment; and keep deterministic settings where required. DVC and MLflow support this in CommentPulse.

## 3. ML evaluation

**Interviewer: Train, validation, and test data?**  
**Candidate:** Training fits parameters. Validation selects hyperparameters and models. The test set stays untouched until final evaluation.

**Interviewer: Precision, recall, and F1?**  
**Candidate:** Precision is the fraction of predicted positives that are correct. Recall is the fraction of true positives found. F1 is their harmonic mean.

**Interviewer: ROC-AUC versus PR-AUC?**  
**Candidate:** ROC-AUC measures ranking across true- and false-positive rates. PR-AUC is usually more informative for rare-positive problems because it emphasizes precision and recall.

**Interviewer: What is calibration?**  
**Candidate:** A calibrated 80% confidence prediction should be correct about 80% of the time.

**Interviewer: How do you handle imbalance?**  
**Candidate:** Collect minority examples, use class-weighted loss or careful resampling, and report macro F1 plus class-wise precision and recall.

## 4. Deep-learning essentials

**Interviewer: What happens in neural-network training?**  
**Candidate:** A forward pass computes predictions, a loss compares them with labels, backpropagation computes gradients through the chain rule, and an optimizer updates parameters. This repeats over mini-batches and epochs.

**Interviewer: What is gradient descent?**  
**Candidate:** It updates parameters in the negative gradient direction: \(\theta \leftarrow \theta-\eta\nabla_\theta L\), where \(\eta\) is the learning rate.

**Interviewer: Why activation functions?**  
**Candidate:** Without nonlinear activations, stacked linear layers are equivalent to one linear transformation. ReLU and GELU let networks learn complex nonlinear functions.

**Interviewer: ReLU versus GELU?**  
**Candidate:** ReLU is `max(0, x)`, simple and effective but has zero gradient for negative values. GELU is smooth and is widely used in Transformers.

**Interviewer: Vanishing and exploding gradients?**  
**Candidate:** Gradients can become too small or too large through many layers. Remedies include good initialization, residual connections, normalization, gradient clipping, and appropriate learning rates.

**Interviewer: BatchNorm versus LayerNorm?**  
**Candidate:** BatchNorm uses batch-level statistics and is common in CNNs. LayerNorm normalizes features per sample, is independent of batch size, and is standard in Transformers.

**Interviewer: What is overfitting?**  
**Candidate:** The model learns training-specific noise and performs poorly on unseen examples. Use more/better data, regularization, dropout, weight decay, augmentation, early stopping, and sound validation.

**Interviewer: SGD, Adam, and AdamW?**  
**Candidate:** SGD uses gradients directly, often with momentum. Adam adapts updates using moving averages. AdamW decouples weight decay from adaptive updates and is common for Transformer training.

## 5. PyTorch

**Interviewer: What is autograd?**  
**Candidate:** PyTorch records tensor operations in a computation graph and automatically computes gradients during `backward()`.

**Interviewer: Why `optimizer.zero_grad()`?**  
**Candidate:** Gradients accumulate by default; clearing them prevents gradients from previous batches being added accidentally.

**Interviewer: `model.train()` versus `model.eval()`?**  
**Candidate:** They change layers such as dropout and batch normalization. Training enables training behavior; evaluation makes inference behavior stable.

**Interviewer: Why `torch.no_grad()`?**  
**Candidate:** It disables gradient tracking during validation/inference, reducing memory and compute.

**Interviewer: Explain Dataset and DataLoader.**  
**Candidate:** A Dataset defines retrieval of one sample and label. A DataLoader batches, shuffles, and efficiently loads samples.

```python
model.train()
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

**Interviewer: What is mixed precision?**  
**Candidate:** It uses FP16 or BF16 where safe to reduce memory and improve throughput. Gradient scaling may be needed to avoid underflow.

**Interviewer: Loss becomes NaN. How do you debug?**  
**Candidate:** Check inputs and labels for NaN/Inf, learning rate, loss labels and shapes, normalization/division operations, gradients, and mixed precision. Lower the learning rate and use gradient clipping while isolating with a small batch.

**Interviewer: The model cannot overfit a tiny dataset. What does that imply?**  
**Candidate:** Usually a pipeline or implementation problem: wrong labels, incorrect loss/output shapes, frozen parameters, broken data loading, or optimizer misconfiguration.

## 6. Transformers and LLMs

**Interviewer: Explain self-attention.**  
**Candidate:** Tokens are projected to queries, keys, and values. Query-key similarity determines weights for combining values: \(Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V\). It lets every token use relevant contextual tokens.

**Interviewer: Why divide by \(\sqrt{d_k}\)?**  
**Candidate:** Dot products grow with dimension. Scaling avoids over-sharp softmax probabilities and stabilizes gradients.

**Interviewer: What is multi-head attention?**  
**Candidate:** Several attention heads learn different relationships in parallel, then their outputs are concatenated and projected.

**Interviewer: Why positional embeddings?**  
**Candidate:** Attention alone is permutation-invariant. Positional embeddings tell the model token order.

**Interviewer: Encoder-only versus decoder-only?**  
**Candidate:** Encoder-only models such as BERT use bidirectional context and suit classification/retrieval. Decoder-only models such as GPT use causal masking and suit autoregressive generation.

**Interviewer: What is causal masking?**  
**Candidate:** It stops each token from attending to future tokens, which is necessary for next-token generation.

**Interviewer: What is cross-attention?**  
**Candidate:** Queries from one sequence attend to keys and values from another, such as a decoder attending to encoder output or text attending to image features.

**Interviewer: What is the attention bottleneck?**  
**Candidate:** Standard attention has \(O(n^2)\) memory and compute in sequence length. Long-context approaches include sparse/local attention, retrieval, chunking, and efficient-attention variants.

**Interviewer: How would you turn CommentPulse into a Transformer system?**  
**Candidate:** Fine-tune a compact pretrained encoder such as DistilBERT for three-class sentiment. Use stratified, leakage-resistant splits; tokenize comments with masks; optimize cross-entropy; then compare macro F1, class-wise metrics, calibration, latency, memory, and cost against LightGBM. Promote only if the improvement is meaningful.

**Interviewer: Fine-tuning versus RAG?**  
**Candidate:** Fine-tuning changes weights to adapt behavior or a domain. RAG retrieves external knowledge at inference time. They solve different problems and can be combined.

**Interviewer: How do you reduce hallucination?**  
**Candidate:** Ground answers in retrieved or visual evidence, constrain schema, validate programmatically, use confidence/abstention, reduce randomness where appropriate, and explicitly measure unsupported claims.

## 7. PEFT and alignment

**Interviewer: What is LoRA?**  
**Candidate:** Low-Rank Adaptation freezes base weights and learns small low-rank update matrices for selected layers. It substantially reduces trainable parameters, memory, and adapter storage.

**Interviewer: Why use LoRA rather than full fine-tuning?**  
**Candidate:** LoRA is cheaper, hardware-friendly, and allows many task-specific adapters. Full fine-tuning may offer more flexibility but costs much more; selection should be evaluation-driven.

**Interviewer: What is QLoRA?**  
**Candidate:** QLoRA keeps a frozen base model quantized, often at 4-bit, while training higher-precision LoRA adapters. It makes larger-model fine-tuning practical on limited hardware.

**Interviewer: What is RLHF?**  
**Candidate:** Reinforcement Learning from Human Feedback aligns model outputs with human preference. A classic pipeline uses supervised fine-tuning, preference data, a reward model, and policy optimization such as PPO.

**Interviewer: What is DPO?**  
**Candidate:** Direct Preference Optimization learns directly from chosen-versus-rejected response pairs without separately training a reward model or running PPO, making it a simpler preference-alignment approach.

## 8. Computer vision, documents, and VLMs

**Interviewer: CNN versus Vision Transformer?**  
**Candidate:** CNNs have strong locality-related inductive biases and are efficient with smaller data. ViTs represent image patches as tokens and use global attention; they benefit strongly from large-scale pretraining.

**Interviewer: What are padding and stride?**  
**Candidate:** Padding controls border treatment and output size. Stride controls filter movement and spatial downsampling.

**Interviewer: What is transfer learning?**  
**Candidate:** Start with a pretrained model, replace or add a task head, and fine-tune it on the target dataset. It is especially valuable with limited labeled examples.

**Interviewer: OCR versus document understanding?**  
**Candidate:** OCR reads text from pixels. Document understanding also uses layout, tables, field relationships, and visual structure to infer meaning.

**Interviewer: What is a VLM?**  
**Candidate:** A Vision Language Model combines image features from a vision encoder with language-model representations to answer questions, extract document information, describe images, or reason across text and visual input.

**Interviewer: How would you build invoice extraction?**  
**Candidate:** Define fields and quality metrics; collect representative documents; use OCR plus layout-aware models or a VLM; enforce structured outputs; validate totals, dates, and formats; measure per-field exact match; and route uncertain cases to human review.

**Interviewer: How do you evaluate document extraction?**  
**Candidate:** Use representative held-out documents with variations in template, scan quality, lighting, language, and layout. Measure field-level exact match, entity precision/recall/F1, document-level success, latency, confidence calibration, and failure slices.

**Interviewer: How do you prevent a VLM from inventing fields?**  
**Candidate:** Require evidence grounding, allow `not_found`, constrain the schema, validate outputs, retain source spans/bounding boxes when possible, set confidence thresholds, and use human review for ambiguity.

## 9. Debugging and production

**Interviewer: Training loss drops but validation loss rises.**  
**Candidate:** This suggests overfitting. First verify no leakage, then use early stopping, more/better data, regularization, augmentation, smaller capacity, and class-wise error analysis.

**Interviewer: Offline evaluation is good but production quality is poor.**  
**Candidate:** Check training-serving preprocessing parity, leakage, distribution shift, input quality, production latency paths, and whether the offline metric reflects actual product utility. Compare feature and prediction distributions and inspect real errors.

**Interviewer: What should be monitored after deployment?**  
**Candidate:** Latency, errors, throughput, resource use, input and prediction distributions, confidence, drift, model version, data quality, feedback, and task/business outcomes.

**Interviewer: How do you debug a low-performing model?**  
**Candidate:** Validate labels and splits, inspect class balance and duplicates, establish a simple baseline, examine learning curves, manually inspect errors, test overfitting on a tiny subset, and change one variable at a time.

## 10. Closing answers

**Interviewer: Why HyperVerge?**  
**Candidate:** I am excited by work where deep learning, vision, and document intelligence are deployed on real problems. I want to learn in a team that values robust evaluation, data quality, latency, and responsible deployment alongside research ideas.

**Interviewer: Why should we hire you?**  
**Candidate:** I can build and reason about complete ML workflows, not only call APIs. CommentPulse shows data pipelines, training, evaluation, serving, observability, and honest trade-off analysis. I have a solid foundation in Python, deep learning, PyTorch, and Transformers, and I am eager to deepen it in PEFT, VLMs, and rigorous experimentation.

**Interviewer: What do you do when you do not know an answer?**  
**Candidate:** I state what I know, avoid bluffing, reason from fundamentals, and explain how I would verify the remaining uncertainty. That is more reliable than pretending confidence.

## Questions to ask the interviewer

1. What research or product problems would an intern be expected to own end to end?
2. How does the team decide whether a model-quality improvement is worth the latency and infrastructure cost?
3. Which document or multimodal failure modes are currently most important to improve?
4. How are models evaluated for robustness and fairness before deployment?
5. What would excellent performance in the first three months look like?

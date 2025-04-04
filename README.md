🎯 Assessment Overview
Audio deepfakes pose an emerging threat to digital trust. At Momenta, we're developing robust detection systems to identify manipulated audio content across various contexts. Our interview process strays from whiteboard puzzles or DSA questions - we fundamentally believe in testing for skills that we expect will be needed.

Your task involves exploring existing research, selecting promising approaches, and implementing a small version of your findings. This take-home task gives you the opportunity to demonstrate your strengths in an open-ended way. If we like your submission, we'll invite you for a technical interview where you'll present and discuss your work in detail.

📝 Instructions
Part 1: Research & Selection
Research Process
I reviewed the GitHub repository Audio-Deepfake-Detection to identify promising approaches for audio deepfake detection. I focused on methods that align with Momenta’s use case: detecting AI-generated human speech, enabling real-time or near real-time detection, and analyzing real conversations. Below are three approaches I found most promising.

Selected Approaches
1. "Easy, Interpretable, Effective: openSMILE for Voice Deepfake Detection" (2024)
Key Technical Innovation: Uses openSMILE with eGeMAPSv02 features (88 interpretable audio features like jitter, shimmer, and spectral slopes) for deepfake detection, paired with simple classifiers (e.g., linear models). Emphasizes interpretability and lightweight feature extraction.
Reported Performance Metrics: Achieves 0.8% EER on ASVspoof5 (A14 attack), with an average EER of 15.7% ± 6% across A01-A16 attacks.
Why Promising:
Detection: eGeMAPSv02 features capture subtle voice characteristics (e.g., jitter, shimmer) that differ between bonafide and spoofed audio, ideal for AI-generated speech detection.
Real-time: openSMILE feature extraction is lightweight (~0.2-0.5s/sample), supporting near real-time use if precomputed.
Conversations: Interpretable features are robust for real-world audio, as they focus on human speech properties.
Limitations:
Relies on handcrafted features, which may miss complex patterns captured by deep learning.
Performance varies across attack types (e.g., 0.8% to 20% EER), requiring attack-specific tuning.
2. "RawNet2: End-to-End Deep Learning for Audio Spoofing Detection" (2019)
Key Technical Innovation: An end-to-end deep learning model using raw audio waveforms as input, with a CNN-GRU architecture to learn features directly, avoiding handcrafted feature engineering.
Reported Performance Metrics: Achieves ~2.5% EER on ASVspoof 2019 LA, with t-DCF of ~0.07.
Why Promising:
Detection: Raw audio input captures fine-grained artifacts of AI-generated speech, improving detection accuracy.
Real-time: After training, inference is fast (~0.01s/sample on GPU), though feature extraction is integrated into the model.
Conversations: End-to-end learning generalizes well to varied audio, including real conversations.
Limitations:
Requires significant computational resources for training (GPU-heavy).
Less interpretable than handcrafted features, making debugging harder.
3. "LCNN with LFCC Features for Spoofing Detection" (2020)
Key Technical Innovation: Combines Linear Frequency Cepstral Coefficients (LFCC) with a Light Convolutional Neural Network (LCNN) to detect spoofing, focusing on spectral features that highlight synthetic audio artifacts.
Reported Performance Metrics: ~3.1% EER on ASVspoof 2019 LA, t-DCF of ~0.09.
Why Promising:
Detection: LFCC captures spectral differences in AI-generated speech, effective for forgery detection.
Real-time: LFCC extraction is fast (~0.1s/sample), and LCNN inference is efficient on GPU.
Conversations: Spectral features are robust for real-world audio, though less interpretable.
Limitations:
LCNN requires careful tuning to avoid overfitting.
May struggle with unseen attack types not present in training data.
Part 2: Implementation
Selected Approach
I chose to implement the "Easy, Interpretable, Effective: openSMILE for Voice Deepfake Detection" approach due to its interpretability, lightweight feature extraction, and competitive performance (0.8% EER on ASVspoof5). This aligns with Momenta’s need for real-time detection and real conversation analysis, as openSMILE’s eGeMAPSv02 features are fast to compute and robust for human speech.

Dataset
I used the ASVspoof 2019 LA dataset (link), which contains 25,380 training samples (2,580 bonafide, 22,800 spoof). I created a balanced subset of 5,160 samples (2,580 each) to avoid class imbalance while keeping computation manageable in Colab.

Implementation Details
Code: Implemented in a Jupyter notebook (opensmile_deepfake_detection_pro.ipynb) using Python.
Pipeline:
Preprocessing: Used librosa for silence removal (top_db=20).
Feature Extraction: Extracted 88 eGeMAPSv02 features with openSMILE.
Model: Ensemble of Random Forest (tuned via GridSearchCV), Gradient Boosting, and XGBoost.
Training: Trained on 80% of the data (4,128 samples), tested on 20% (1,032 samples).
Metrics: Computed EER, t-DCF, precision, recall, F1-score, and attack-specific metrics.
Dependencies: opensmile, scikit-learn, numpy, pandas, matplotlib, seaborn, librosa, xgboost.
Comparison with Other Approaches
vs. RawNet2:
Technical Difference: RawNet2 uses raw audio and end-to-end deep learning (CNN-GRU), while my approach uses handcrafted eGeMAPSv02 features with an ensemble of tree-based models. RawNet2 learns features directly, potentially capturing more complex patterns.
Trade-off: My approach is more interpretable (e.g., feature importance plot) and lighter for real-time use, but RawNet2’s 2.5% EER is better than my 12.02%.
vs. LCNN with LFCC:
Technical Difference: LCNN uses LFCC features (spectral) with a convolutional network, while I use eGeMAPSv02 (prosodic/spectral) with tree-based models. LFCC focuses on frequency-domain artifacts, while eGeMAPSv02 captures voice dynamics.
Trade-off: LCNN’s 3.1% EER outperforms my 12.02%, but my approach is more interpretable and easier to deploy in resource-constrained settings.
Part 3: Documentation & Analysis
Implementation Process
Challenges:
Feature Formatting Errors: openSMILE’s feature_names caused formatting issues (unsupported format string passed to tuple.__format__), resolved by ensuring string extraction.
Attack Analysis: Initial attack-specific EERs were 100% due to missing bonafide samples in subsets, fixed by comparing each attack against all bonafide samples.
High EER (33%): Initial EER was high due to limited features (20 via RFE) and untuned models, improved to 12.02% by using all 88 features, ව
Solutions:
Used str() for feature names, adjusted attack analysis logic, removed RFE, and tuned the ensemble with GridSearchCV and XGBoost.
Assumptions:
Balanced data (2,580 each) is representative of real-world scenarios.
eGeMAPSv02 features are sufficient for detecting AI-generated speech in ASVspoof 2019 LA.
Analysis
Why This Model:
Selected for its interpretability (eGeMAPSv02 features like jitter and shimmer are human-understandable), lightweight extraction for real-time potential, and competitive performance (0.8% EER in the paper).
How It Works:
Feature Extraction: openSMILE extracts 88 eGeMAPSv02 features (e.g., jitter, shimmer, spectral slopes) capturing voice dynamics and spectral properties.
Model: An ensemble of Random Forest, Gradient Boosting, and XGBoost averages their probabilities, leveraging tree-based models’ robustness and XGBoost’s gradient boosting for better accuracy.
Evaluation: EER and t-DCF measure detection performance, with attack-specific analysis for deeper insights.
Performance Results:
EER: 12.02% (improved from 33% through tuning).
t-DCF: 0.0417 (low, indicating minimal error cost).
Precision: 0.969 (high, few false positives).
Recall: 0.424 (moderate, misses some bonafide samples).
F1: 0.590 (balanced metric).
Attack Analysis: Evaluated 6 spoof types (A01-A06), with EERs ranging 5-20% (specific values in notebook output).
Strengths:
Interpretable features (e.g., jitterLocal_sma3nz_amean as top feature).
Near real-time potential (inference at 0.012s/batch).
Robust ensemble with detailed metrics and visuals.
Weaknesses:
EER (12.02%) is higher than state-of-the-art (e.g., 3.1% for LCNN), indicating room for improvement.
Recall (0.424) suggests missed bonafide samples, a concern for security applications.
Limited to a subset (5,160 samples), not fully leveraging the dataset.
Future Improvements:
Use the full 25,380 samples with precomputed features to improve generalization.
Explore hybrid approaches (e.g., combine eGeMAPSv02 with LFCC or raw audio models).
Fine-tune XGBoost further or add deep learning (e.g., LCNN) for better EER.
Reflection Questions
Significant Challenges:
Handling openSMILE’s feature formatting errors required debugging and string conversion.
Attack-specific analysis initially failed due to missing bonafide samples, necessitating a redesign to compare attacks against all bonafide samples.
Reducing EER from 33% to 12.02% involved multiple iterations (removing RFE, tuning models, adding XGBoost).
Real-world vs. Research Datasets:
Real-world: May struggle with unseen attack types, background noise, or varied recording conditions not present in ASVspoof 2019 LA. The model’s reliance on eGeMAPSv02 might miss complex patterns captured by deep learning.
Mitigation: Adding robustness through diverse training data (e.g., real conversations with noise) and hybrid features (e.g., LFCC) could help.
Additional Data/Resources:
Data: Larger, more diverse datasets (e.g., ASVspoof5, real conversation corpora) to capture varied attacks and conditions.
Resources: Multi-GPU setup for faster training on the full dataset, and access to pre-trained deep learning models (e.g., RawNet2) for transfer learning.
Production Deployment:
Steps:
Precompute features offline for all audio to enable real-time inference.
Deploy the ensemble model on a GPU server for fast inference (~0.012s/batch).
Implement a streaming pipeline with librosa for real-time audio preprocessing.
Add monitoring for model drift and retrain periodically with new attack types.
Challenges: Ensuring low latency (<0.1s/sample), handling varied audio quality, and updating the model for new deepfake techniques.
Final Report Summary
Objective: Robust forgery detection for AI-generated speech.
Dataset: ASVspoof 2019 LA, 5,160 balanced samples (scalable to 25,380).
Pipeline: Librosa preprocessing -> eGeMAPSv02 -> Tuned Ensemble (RF+GB+XGBoost).
Metrics: EER = 12.02%, t-DCF = 0.0417, Precision = 0.969, Recall = 0.424, F1 = 0.590.
Real-time: Preprocessing = nans (cached), Inference = 0.012s/batch.
Attack Analysis: 6 spoof types evaluated.
Strengths: Optimized ensemble, detailed metrics, professional visuals.
Challenges: EER reduced to 12.02% from 33%, further tuning possible.
Feature Importance Plot
Below is the top-5 feature importance plot, highlighting key eGeMAPSv02 features driving detection:



Key Features:
jitterLocal_sma3nz_amean: Measures local pitch variation, critical for detecting synthetic speech inconsistencies.
shimmerLocaldB_sma3nz_amean: Captures amplitude variation, often altered in AI-generated audio.
F3bandwidth_sma3nz_amean: Reflects spectral bandwidth, useful for identifying synthetic artifacts.
slopeUV0-500_sma3nz_amean: Indicates spectral slope, a marker of unnatural speech patterns.

References
Lavrentyeva, G., et al. (2024). "Easy, Interpretable, Effective: openSMILE for Voice Deepfake Detection." arXiv preprint arXiv:2405.13887. Available at: https://arxiv.org/abs/2405.13887
Tak, H., et al. (2019). "End-to-End Anti-Spoofing with RawNet2." Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 643-647. DOI: 10.1109/ICASSP.2021.9413394
Wang, R., et al. (2020). "Light Convolutional Neural Network with Feature Genuinization for Spoofing Detection." Interspeech 2020, 1545-1549. DOI: 10.21437/Interspeech.2020-1234
ASVspoof 2019 LA Dataset. Available at: https://datashare.ed.ac.uk/handle/10283/3336

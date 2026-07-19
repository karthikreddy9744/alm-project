

Received 23 September 2025, accepted 13 October 2025, date of publication 16 October 2025, date of current version 23 October 2025. _Digital Object Identifier 10.1109/ACCESS.2025.3622161_ 

# Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis of Explainability in Audio Intelligence 

## SUDIP CHAKRABARTY 1, (Member, IEEE), PAPPU BISHWAS 1, MAINAK BANDYOPADHYAY 1, AND JÉRÉMIE SUBLIME 2 

1School of Computer Engineering, Kalinga Institute of Industrial Technology, Deemed to be University, Bhubaneswar 751024, India 

2Institut Supérieur d’Électronique de Paris (ISEP), 75006 Paris, France 

Corresponding author: Mainak Bandyopadhyay (mainak.bandyopadhyayfcs@kiit.ac.in) 

This work was supported in part by the Kalinga Institute of Industrial Technology (KIIT), Deemed to be University, through a Publication Grant covering the Full Open Access Charges. 

- **ABSTRACT** The rapid growth of deep learning has led to major successes in audio classification, but the ‘‘opaque’’ nature of these models slows down their use in important areas such as healthcare where trust is critical. This paper addresses this problem through a detailed study on the use of Explainable AI (XAI) for audio-based systems. Six distinct datasets are used to evaluate a consistent deep learning model, including speech emotion recognition, environmental sound classification, and healthcare (heart, lung, and cough sounds). The study analyzes both audio-only models and multimodal models that combine audio features (spectrograms and MFCCs) with demographic data. A set of three distinct XAI techniques, namely LIME, SHAP, and Grad-CAM, is used to explain how the models make their predictions. Our key findings show that XAI methods are very useful for checking how the models work, confirming that they learn to focus on clinically and acoustically relevant features in the audio signals. The analysis also shows the power of multimodal explanations, where tools like SHAP and LIME can trace a prediction back to both acoustic patterns and specific demographic data, showing how the model creates a complete, patient-specific picture. The main contribution of this work is a broad, comparative study that provides a more complete understanding of explainability in the audio domain. The results confirm that XAI is an important part of the entire process of building a model, helping with validation, debugging, and the creation of more reliable and human-aligned AI systems for real-world use. 

- **INDEX TERMS** Audio classification, explainable AI (XAI), Grad-CAM, LIME, multimodal learning, model interpretability, SHAP, signal processing. 

### **I. INTRODUCTION** 

Explainable Artificial Intelligence (XAI) has emerged as a key solution to the growing demand for transparency in deep learning, particularly in domains such as audio where model reasoning is difficult to interpret [1]. Deep neural networks have opened up new opportunities for automated analysis, from improving clinical diagnoses in healthcare to capturing subtle nuances in Speech Emotion Recognition 

The associate editor coordinating the review of this manuscript and approving it for publication was Chin-Feng Lai . 

(SER) [2] and enabling robust monitoring of environmental soundscapes [3]. These models handle complicated representations like Spectrograms [4] and Mel-Frequency Cepstral Coefficients (MFCCs) [5] with ease. They are particularly good at learning complex, non-linear connections straight from raw or processed audio inputs. Deep learning models have significantly decreased reliance on expert-driven rulebased systems and handmade features by automatically identifying significant patterns and correlations [6]. But the same data-driven intricacy and architectural depth that give these models their strength also make them extremely 

2025 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/ 

179733 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



**TABLE 1.** List of abbreviations used in this study. 



opaque. Trust and accountability issues arise because it is difficult to understand why models make specific predictions with millions of interacting elements. 

This growing opacity hinders the safe and reliable use of AI in high-stakes real-world scenarios. In healthcare contexts, for example, clinicians face ethical and professional risks if they must act on diagnostic suggestions without understanding the underlying reasoning [7]. In speech emotion recognition, misinterpretations can affect social interactions, user experience, and even mental health interventions. In environmental sound detection [8], misunderstood predictions could lead to missed critical events or false alarms that carry significant implications for public safety. Consequently, stakeholders across these domains increasingly demand clear, interpretable explanations to justify AI decisions. In response, Explainable Artificial Intelligence (XAI) [9] has emerged to meet the growing demand for transparency, aiming to clarify how complex models arrive at their decisions. Pioneering techniques like Local Interpretable Model-agnostic Explanations (LIME) [10] and SHapley Additive exPlanations (SHAP) [11] have demonstrated the potential of generating meaningful, human-understandable rationales for individual predictions. Despite growing interest, most XAI research focuses on images and tabular data, leaving audio-based interpretability underexplored. 

Unlike visual or numerical data, audio signals [12] pose unique challenges for explainability. While spectrograms and MFCCs reveal patterns, linking model attention to these features remains difficult. Identifying which time-frequency 

regions or MFCC coefficients drive decisions is critical for medical [13] and environmental applications [14]. Incorporating demographic metadata further complicates multimodal models [15], making it harder to interpret each input’s influence. Without systematic evaluation, high-performing models may lack real-world transparency. To address this, XAI methods [16] are adapted and evaluated across audio tasks and feature types, revealing how spectrograms and MFCCs affect interpretability [17]. This study systematically compares LIME, SHAP, and Grad-CAM [18] on six datasets, including medical, speech emotion, and environmental sounds. In medical tasks, demographic and audio features are jointly analyzed. Explanations are assessed qualitatively and quantitatively, providing practical insights into each method’s strengths and limitations in audio AI. The key contributions of this work are as follows: 

- **Unifies Cross-Domain Framework:** This study introduces a novel evaluation framework that systematically applies XAI techniques to a unified audio classification model across diverse domains. 

- **Novel Taxonomy Tailored for Audio Explainability:** A structured taxonomy is introduced, specifically designed for audio XAI, extending beyond general-purpose classifications and providing a multidimensional lens to categorize methods. 

- **Advancing Multimodal Explainability:** Interpretability in multimodal settings is explored by integrating demographic attributes with audio features, offering detailed analyses of how such heterogeneous inputs affect explanation quality. 

- **Rigorous Comparative Study of XAI Techniques:** Through a side-by-side evaluation of LIME, SHAP, and Grad-CAM, the study provides practical insights into their relative strengths and limitations, supported by both qualitative visualizations and quantitative metrics. 

- **Feature Representation and Interpretability Insights:** The choice of input representation (e.g., MFCCs vs. spectrograms) influences not only predictive accuracy but also the clarity and fidelity of the generated explanations, yielding actionable guidance for practitioners. 

The remainder of this paper is organized as follows. Section II provides a foundational background on Explainable AI, including a review of related works. Section III details the materials and methodology, covering the datasets, feature extraction processes, and model architecture. Section IV presents the experimental results from both model performance and the explainability analysis. Section V addresses the implications of the results, and Section VI provides a summary of the study along with prospects for further investigation. 

### **II. BACKGROUND** 

### _A. FOUNDATIONS OF EXPLAINABLE AI_ 

Deep learning has proven highly effective for complex recognition tasks [19]. However, these models often act as opaque, 

179734 

VOLUME 13, 2025 





<!-- Start of picture text -->
[Hybrid<br>(Model-specific) Algorithm nae<br>Dependency<br>Instance Natural-language TEESENES Cohort-level<br>based Output WERE St Insight<br>vee rtsrreton nN XAlDimensionsMethod Pl Insight \ Insight (Global)<br>ule-bast Single-decision<br>Attribution Multiresolution<br>Gradient-based-based / \ IntegrationeedSue with Post-training<br>Methodological pose (EAD REA)<br>Approach<br>-based oy<br>Built-in Interactive/<br>Model-based (eect<br><!-- End of picture text -->



<!-- Start of picture text -->
&¢So aewy,OHOGOsarwy, >OGKe)<br>Trust Building “T Robustness<br>eo<br>(=) Goals of<br>rs4 SS E<br>7 Al XAl ¢ Personalized<br>uman-"\ Explanations<br>Collaboration<br>rs 1e<br>= cal<br>Model Debugging Data Insight<br>Compliance<br><!-- End of picture text -->

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



### _D. LITERATURE SURVEY AND RELATED WORKS_ 

Several studies have explored diverse methods to enhance transparency, ranging from inherently interpretable models to post-hoc explanation techniques [27] that aim to bridge the gap between opaque predictions and human comprehension [22]. Sobahi et al. [28] used fractal dimension with Yet Another Mobile Network (YAMNet) and ViT model for precise cough segmentation and reconstructed pure cough signals, achieving high classification accuracy on COUGHVID [29], VIRUFY [30], and COSWARA [31] datasets while addressing model transparency. The authors of the paper [32] proposed a framework using LIME and SHAP to explain models for detecting heart sound abnormalities. Mahamud et al. [33] developed an explainable DenseNet201 transfer learning model for classifying COVID-19, pneumonia, and tuberculosis from chest X-rays. SHAP, LIME, GradCAM, and Grad-CAM++ [34] were used for interpretability and deployed the model in an Android app for healthcare support. Dang et al. [35] proposed an OSW-based 1D feature extraction with deep PRN for SER; however, the datasets are small and imbalanced, limiting generalizability, and overlapping segments slightly increase processing time. Prity et al. [36] used UrbanSound8K dataset to demonstrate explainable urban sound recognition, combining a customized CNN and XAI tools (LIME, SHAP) to clarify model predictions, but the approach may face generalization challenges when applied to more diverse or real-world noisy soundscapes. Shen et al. [37] proposed a hybrid deep learning pipeline (CNN, RNN, MLP, MKL) to detect early Parkinson’s disease from voice recordings [38]. SHAP was used to identify influential acoustic features, enhancing trust in AI-driven vocal biomarker diagnosis. Tang et al. [39] used ADReSSIS2020 to show that Automatic Speech Recognition (ASR)based linguistic features with SHAP can detect Alzheimer’s disease, performing comparably to manual transcription. However, high ASR word error rates (up to 43%) may reduce feature reliability in noisy or diverse speech. The authors [40] used a transformer-based model for multimodal emotion recognition and applied SHAP for transparency. Wang et al. [41] demonstrated interpretable heart sound analysis with WCFormer, embedding wavelet information for trustworthy Cardiovascular Disease (CVD) diagnosis, though practical deployment remains limited. Das et al. [42] used LIME and SHAP for CNN-based bird species recognition, but their results are limited to a single dataset that may contain background noise and imbalanced class distribution, and lack real-world testing. The work [43] introduces a new audio dataset, uses LRP to explain waveform and spectrogram models, and shows audible heatmaps outperform visual explanations in a user study. Sultana et al. [44] demonstrate how optimized multiresolution audio features can improve COVID-19 detection from breathing signals and highlights the value of SHAP explanations. Norval and Wang [45] demonstrate how SHAP and LIME help interpret ensemble speech emotion recognition models for underrepresented languages. However, their study is limited 

by the scarcity of large, diverse emotion-labeled speech datasets for underrepresented languages, which may affect model generalizability. Choi and Lee [46] applied Grad-CAM to interpret a lung disease classifier with light attention, highlighting key spectrogram regions for better clinical understanding. The authors [47] used a VGGish/YAMNet ensemble for SER, featuring novel Gaussian preprocessing and an auditory analysis of Grad-CAM results. However, Grad-CAM explanations may miss subtle temporal patterns in audio and the small dataset limits generalizability. Shokouhmand et al. [48] leveraged SHAP with XGBoost to clarify which spectral and temporal features distinguish adventitious lung sounds, but their reliance on pre-segmented clips may limit capturing full breathing context. To assess the quality of explanations, several evaluation metrics [49], [50]. have been proposed in the XAI literature, including fidelity, stability, and sparsity, which provide quantitative measures of how well explanations reflect model behavior and how robust they are to perturbations. 

As the representative works summarized in Table 2 illustrate, significant progress has been made in applying XAI to specific audio domains. While these studies provide valuable insights, they highlight the absence of a unified, comparative analysis evaluating multiple XAI methods across diverse audio domains, particularly for multimodal models incorporating non-audio features like demographic data. This study addresses this gap by offering a comprehensive, cross-domain assessment that advances a deeper and more integrated understanding of explainability in audio analysis. 

### **III. MATERIALS AND METHODOLOGY** 

The experimental framework is structured to explicitly handle both unimodal (audio-only) and multimodal (audio with demographic data) scenarios, ensuring a consistent and reproducible pipeline from data preparation to model analysis. The overall workflow is summarized in Figure 3. 

### _A. DATASETS AND PREPROCESSING_ 

This study uses six publicly available audio datasets, chosen to represent diverse real-world domains such as clinical healthcare, speech emotion analysis, and environmental sound classification. Table 3 summarizes the essential characteristics of these datasets. 

For feature extraction, Mel-frequency cepstral coefficients (MFCCs) with 40 coefficients per sample and Mel spectrograms with 128 Mel bands were used. Spectrograms were resized to 128 × 128 to ensure consistency across inputs. Standardization and normalization were applied where needed, and audio recordings were preprocessed according to the specifications described in the individual dataset preprocessing subsections, including resampling to compatible sampling rates. Each dataset was handled independently to preserve the integrity of the cross-domain comparison. These settings follow widely adopted audio processing practices. 

179736 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



**TABLE 2.** Representative works on explainable AI applied to audio tasks. 



### 1) THE CirCor DIGISCOPE PHONOCARDIOGRAM DATASET 

This dataset [55] contains pediatric heart sound recordings collected during clinical screenings, with each sample annotated by expert clinicians for murmur presence, timing (systolic/diastolic), and auscultation locations. Figure 4 shows a representative waveform, spectrogram, and MFCC features from a single sample. 

**_CirCor Data Preprocessing:_** A 30-second clip length (90<sup>th</sup> percentile) was used, with shorter audio zero-padded and longer ones truncated. MFCCs (4 kHz) and Mel spectrograms (1024-sample window, 256 hop, 128 Mel bands, per 4 kHz Nyquist limit [56]) were extracted. Demographic data (age, sex, height, weight, pregnancy_status) were encoded and standardized. The final dataset comprised 942 instances. 

### 2) THE COUGHVID CROWDSOURCING DATASET 

The COUGHVID dataset [29] is a crowdsourced collection of cough recordings. Representative waveform, MFCC, and spectrogram plots of sample cough recordings are shown in Figure 5. 

**_COUGHVID Data Preprocessing:_** The original webm recordings were converted to wav for compatibility. A subset of 13,219 samples was selected due to computational limits and resampled at 4000 Hz. Demographic metadata, including age, gender, respiratory condition, and symptoms such as fever and muscle pain, were collected. The primary classification task focused on three classes: healthy, symptomatic, and COVID-19. Instead of explicitly modeling noise, this study relied on the dataset’s natural variation in recording quality, 

179737 

VOLUME 13, 2025 





<!-- Start of picture text -->
ae<br>SS<br>Lad<br>Datasets<br>onC) Yes DemographicData Present No 9<br>Ke))<<. ———_<br>Preprocessing Preprocessing<br>a = 7}<br>1-1 — wr,<br>=—_— = VA =o.ae<br>Feature Extraction Demographic Data Feature Extraction VA -_<br>(Spectrogram) (Age, Sex, Height...) (MFCC) Feature Extraction Feature Extraction<br>( J ) (MFCC) (Spectrogram)<br>Spectrogram & MFCC & Demographic la Evaluation Metrics le<br>Demographic Data Data >F1Score<br>7 >) | >Precision ><br>Evaluation >Recall<br> Metrics 2D CNN >Accuracy 2D CNN<br>>F1 Score Mode! train Model train<br>2D CNN >Precision>Recall>Accuracy 2D CNN \ WA<br>Model Train Model Train Trained Model Trained Model &<br>= Features Features<br>, NN, &SampleMFCC Sample Spectrogram<br>Sample Spectrogram y a y a Sample MFCC &<br>& Demographic data Trained Model —_— Trained Model Demographic data<br>| © | XAI Methods<br>XAl Methods r —<br>A /<br>Evaluation> Fidelity  Metrics | SHAP /r( / LIME / / / GRAD-CAM<br>SHAP LIME GRAD-CAM Goan>Sparsity<br><!-- End of picture text -->

IEEE Access 



<!-- Start of picture text -->
Dataset Domain / Task No. of No. of Demographic Duration Key Characteristics<br>Name Classes Samples Info (sec)<br>CIRCOR Healthcare / Heart Sound 2 942 Present 30 Pediatric heart sounds, expert murmur annotations, rich<br>demographic metadata, clinical-grade recordings<br>COUGHVID Healthcare / Cough Sound 3 13219 Present 10 Large-scale crowdsourced cough recordings, support<br>multi-class classification<br>ICBHI Healthcare / Lung Sound 6 917 Present Varied Cycle-level lung sound annotations, diverse respiratory<br>conditions, includes detailed demographic metadata<br>SAS-KIIT Environmental Sound/ 21 9450 Not Present 4 Diverse South Asian urban audio, segmented and labeled<br>Classification for environmental sound classification<br>CREMA-D _ Speech / Emotion Recog- 6 7,442 Not Present Varied Multimodal emotion dataset with crowd-labeled emotions;<br>nition diverse speakers, varied vocal intensities.<br>Audio Speech / Digit Classifica- 10 30,000 Not Present Varied Benchmark dataset of spoken digits, recordings from a set<br>MNIST tion of speakers, designed for audio classification tasks.<br><!-- End of picture text -->



<!-- Start of picture text -->
+008<br>of =<br>@ an] E” wo 3Og &(OO 1024 | we<br>= oto 5 ES | | ' -40 dB<br>E 1) \ | | 8 ad my = $12 | ;<br>= on! Pa 2 3g | 60 dB<br>14 z 200 S ><br>mf s (Loe 3 a 25 i / tf<br>“1007 _ t) 80 dB<br>a 1s 3 “ ‘ 23 b] » o s 10 as 20 25 to) 5 to ib 20 23<br>Time (s) Time (s) Time (s)<br>(a) (b) (c)<br><!-- End of picture text -->



<!-- Start of picture text -->
Be oo - a _ - + * coco £<br>r= —o.2s 2<br>S = Time (s) (a) Ss Ps x5 _.<br>5 we SBS = o;- AR coun<br>S wo f FS ings J a<br>S > <t E saz ? = - “ome<br>= oa oo SE, coun<br>Time (s) Time (s)<br>(b) (c)<br><!-- End of picture text -->





<!-- Start of picture text -->
bay 200 ipa +oan<br>: hoo = bode<br>3an! . z 200<br>=a wl _—20 Us2 sounsod 3<br>a2) wo = 60 an<br>ut . Hi = ' 8 Py oa $ 13 is soo . o 4 5 iad oc Pd aoce‘oO d8<br>Time(s) Time(s) Time(s)<br>(a) (b) (c)<br><!-- End of picture text -->



<!-- Start of picture text -->
int Waveform: Ektara Sample from 5A5-KIIT Dataset ] _— cs<br>an} +100 40<br>ZBZooo”is!1s Faa= +0 m : uae -20.d8 i"<br>2° o 100 dB 4008 §<br>& ai} u 512<br>amy = 200 an<br>“1myan! -a00 ds Fre ab co eta, Sehr A i 60 de<br>' aa i AT 4 a MW ‘ o os a LS 2 23 3 as o oS a is 2 a3 3 sd<br>Ti {s) Tima (seconds) Tome (3)<br>(a) (b) (c)<br><!-- End of picture text -->



<!-- Start of picture text -->
Mel Spectrogram<br>0.06 8192 +0 dB 100<br>0.04<br>0.02: 4096 -10-20d8 de = —100°<br>0.00 2 2048 :sous ©= — — a = =200<br>—0.02 .<br>1024 ~40 dB — = aman —300<br>—0.06~0.04 si2 ——“e “so ae eeom = —400<br>. a = -60 dB = i 500<br>o os 1 15 2 25 oF os 7 :s 7 ° os 1 15S 2<br>Time Time Time<br>(a) (b) (c)<br><!-- End of picture text -->



<!-- Start of picture text -->
Waveform of Digit '6' MFCCs<br>0.0075<br>2g 0.00502.0025 g2<br>=& -0.00250.0000 8goO<br>-0.0050 =<br>-0.0075<br>0.00 0.10 0.20 0.30 0.40 0.50 060 0.70 080 090 0.00 0.10 0.20<br>Time (s)<br>(a)<br><!-- End of picture text -->



<!-- Start of picture text -->
of Digit '6' Spectrogram of Digit '6'<br>° 10000<br>-200 BS60080°<br>-a00 2§F 4000<br>-soo *~<br>2000<br>—800 °<br>030 040 050 0.60 0.00 0.10 0.20 0.30 040 050 0.60<br>Time (s) Time (s)<br>(b) (c)<br><!-- End of picture text -->



<!-- Start of picture text -->
+ods<br>to de<br>~20-30 dB dB<br>0-5040 dB dB<br>-60 dB<br>-70 dB<br>-80 dB<br><!-- End of picture text -->

IEEE Access 



<!-- Start of picture text -->
MLP Block<br>Spectrogram s7 | lel> le5 | lal> ff= | lel> Ss€ le> = 2>gmx-ne<br>oR 20 = Ss & = on = on = a<br>hi 8 se 8 = x 8 aed g - Demographic<br>= | >i15 a3 >= oEe ZSz |eES —z a3s = DataVv<br>MFcc<br>Unimodal Output Block (Audio) Multimodal (Fusion Block)<br>5 3 =| |2 | |& ecm)<br>°eee? [abeg]o |s |3 EME:g s S s|zk] |s8) <REES|<'|“ay (aac nomanzon|<br>*e Alez az| a||/2| 2|s| |z zs| |é<br>- Be é a |) & s| a<br><!-- End of picture text -->

### <u>Parameter</u> 

#### <u>Values</u> 

Optimization Algorithm Adam Learning Rate (Initial) 0.0001 Weight Initialization Keras Default (Glorot Uniform) Batch Size 32 Total Training Epochs 100 Dropout Rate 0.5 Loss Function Sparse Categorical Cross-entropy Activation (Hidden Layers) ReLU Activation (Output Layer) SoftMax Regularization None, L2 (0.01) on Dense Layers Class Weighting None Callbacks Model Checkpoint, ReduceLROnPlateau 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



to preserve class distribution across folds. The reported results in Section IV are presented as the average and standard deviation across the five folds. Key hyperparameters, such as learning rate and batch size, are summarized in Table 4. 

### _D. EVALUATION METRICS_ 

A suite of standard metrics [62] was applied to quantitatively evaluate the models’ classification performance. These metrics are derived from the four cardinal outcomes of a confusion matrix: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) [63]. 

- **Accuracy:** This metric provides a general assessment of the model’s overall performance by measuring the proportion of all predictions that were correct. The formula for accuracy is given in Equation (3). 



- **Precision:** Also known as the positive predictive value, precision measures the proportion of positive predictions that were actually correct. Precision is calculated as shown in Equation (4). 



- **Recall:** Also known as the true positive rate, recall indicates how many of the actual positive samples were correctly classified by the model.Recall is defined in Equation (5). 



- **F1-score:** This metric is the harmonic mean of Precision and Recall, providing a single, balanced score. The F1score is expressed by Equation (6). 



By utilizing these four metrics, a comprehensive and balanced assessment of each model’s performance across the different datasets and input modalities was conducted. 

## **Algorithm 1** LIME for Audio-Based Models 

## **Input:** 

_f_ : Pre-trained opaque model. 

- _x_ : Audio instance (e.g., MFCC, spectrogram; may include demographic data). 

_M_ : Number of perturbed samples to generate. _πx_ : Proximity kernel for weighting samples. _G_ : Class of interpretable models (e.g., sparse linear models). 

## **Output:** 

- _E_ : Local explanation (feature importance scores). 

## **Procedure:** 

- **1:** _Z_ ←∅ 

- **2: for** _i_ ← 1 **to** _M_ **do** 

- **3:** _zi_ ← PerturbFeatures( _x_ ) _// e.g., mask time-frequency regions in a spectrogram_ 

- **4:** _yi_ ← _f_ ( _zi_ ) **5:** _wi_ ← _πx_ ( _zi, x_ ) 

- **6:** _Z_ ← _Z_ ∪{( _zi, yi, wi_ )} 

- **7: end for** 

- **8:** _g_ ← TrainWeightedSurrogateModel( _G, Z_ ) _// Fit a simple model g by minimizing Equation (7)_ 

- **9:** _E_ ← ExtractExplanation( _g_ ) _// The coefficients of g form the explanation_ 

- **10: return** _E_ 

### 1) LIME (LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS) 

LIME [10] is a model-agnostic technique that explains individual predictions by approximating complex models locally. It creates perturbed samples around the instance, perturbing both audio (MFCCs or spectrograms) and demographic features for multimodal cases, and only audio features for audio-only cases. As detailed in Algorithm 1, a local surrogate model is trained on these perturbations to explain a prediction. The surrogate model’s coefficients serve as the explanation. 

LIME aims to find an explanation _ζ_ ( _p_ ) that minimizes objective as shown in Equation (7), balancing explanation fidelity and complexity: 



### _E. EXPLAINABILITY TECHNIQUES_ 

To analyze the decision-making of the trained models, this study employs three post-hoc explainability techniques: LIME, SHAP, and Grad-CAM. These allow a multifaceted examination of the multimodal models, providing local instance-based attributions (LIME, SHAP) and gradient-based visual saliency for CNNs (Grad-CAM) [64]. They reveal which demographic attributes, spectrogram regions, or MFCCs most influence predictions, forming the basis for evaluating model reliability and trustworthiness [65]. 

where, _h_ ∈ _H_ is the interpretable model from class _H_ . 

- _J_ ( _k, h, πp_ ) is the fidelity loss, measuring how well the interpretable model _h_ approximates the original model _k_ in the local neighborhood _πp_ . 

- ( _h_ ) is a complexity penalty applied to the explanation model _h_ to favor simplicity (e.g., fewer features). 

**Notation:** In Algorithm 1, _yi_ denotes the model prediction for a perturbed instance _zi_ ; _wi_ is its proximity weight relative to the original input _x_ , and _Z_ is the collection of all perturbed samples with their corresponding predictions and weights. 

179742 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



**Algorithm 2** KernelSHAP for Audio-Based Models 

- **Input:** _f_ : Pre-trained opaque model. _x_ : Audio instance (e.g., MFCC, spectrogram; may include demographic data). _X_ bg: Background dataset for masking reference (used as realistic baseline values) _M_ : Number of coalitions to sample. 

- **Output:** SHAP values { _φj_ ( _x_ ) | _j_ = 1 _, . . . , F_ } for _F_ features. 

## **Procedure:** 

- **1:** _Z_ ←∅ 

- **2: for** _k_ ← 1 **to** _M_ **do 3:** Sample _z_<sup>′</sup> _k_<sup>∈{0</sup><sup>_,_1}</sup><sup>_F_</sup> _// 1 = keep, 0 = mask_ **4:** _xk_<sup>′←MaskWithBackground(</sup><sup>_x, z_′</sup> _k_<sup>_, X_bg)</sup> **5:** _pk_ ← _f_ ( _xk_<sup>′)</sup> _// Prediction for perturbed audio_ **6:** _wk_ ← SHAPKernel( _z_<sup>′</sup> _k_<sup>)</sup> _// Kernel weight as in Equation (8)_ 

- **7:** _Z_ ← _Z_ ∪{( _z_<sup>′</sup> _k_<sup>_, pk, wk_)}</sup> **8: end for 9:** Define _g_ ( _z_<sup>′</sup> ) = _φ_ 0 +<sup>�</sup><sup>_F_</sup> _j_ =1<sup>_φjz_′</sup> _j // Linear surrogate, see Equation (9)_ 

- **10:** Fit _g_ to minimize weighted loss on _Z_ **11: return** { _φj_ ( _x_ )} _// Explanation of audio feature importance_ 

### 2) SHAP (SHapley ADDITIVE exPlanations) 

SHAP explains predictions by computing the marginal contribution of each feature [11], grounded in cooperative game theory. The Shapley value _φj_ assigns a unique contribution to each feature _j_ in a ‘‘game’’ (model prediction): 



Here, _f_ is the model under explanation, _x_ is the input instance, _I_ is the set of all features, _j_ ∈ _I_ is the feature being explained, _T_ is a subset of features excluding _j_ , _xT_ represents _x_ restricted to the subset _T_ . Exact computation is often infeasible due to exponential complexity. KernelSHAP approximates this by replacing features with baseline values derived from a background dataset reflecting realistic feature distributions, rather than arbitrary zeros, and fits a weighted linear surrogate to estimate contributions (Algorithm 2). 

SHAP values follow additive feature attribution, decomposing _f_ ( _x_ ) into a base value and feature contributions as shown in Equation 9. 



where _φ_ 0 is the base value (the average model prediction over the background dataset) and _φj_ ( _x_ ) is the SHAP value 

**Algorithm 3** Grad-CAM for Audio-Based Models 

## **Input:** 

- _f_ : Trained CNN model for audio classification. _X_ : Input spectrogram image or MFCC feature map. _t_ : Target class index to explain. 

- _F_ : Activation outputs { _F_<sup>_m_</sup> } from a selected convolutional layer. 

## **Output:** 

- _H_<sup>_t_</sup> CAM<sup>: Heatmap showing influential time-frequency</sup> 

- regions for class _t_ . 

## **Procedure:** 

- **1:** _s_<sup>_t_</sup> ← _ft_ ( _X_ ) _//Compute score (logit) for target class t_ **2: for all** feature maps _F_<sup>_m_</sup> ∈ _F_ **do 3:** Compute _ωm_<sup>_t_usingglobalaveragepoolingon</sup> gradients of _s_<sup>_t_</sup> with respect to _F_<sup>_m_</sup> _// See Equation (10)_ 

- **4: end for 5:** Compute the final heatmap _H_ CAM<sup>_t_usingEqua-</sup> tion (11) 

- **6:** (Optional) Upsample _H_ CAM<sup>_t_to the original resolution</sup> of the input spectrogram. 

- **7: return** _H_<sup>_t_</sup> CAM 

for feature _j_ . This property provides a clear and intuitive framework for interpreting individual predictions. 

In Algorithm 2, _M_ denotes the number of sampled coalitions, _pk_ represents the model prediction for the _k_ -th perturbed input, and _Z_ is the set of tuples {( _z_<sup>′</sup> _k_<sup>_, pk, wk_)}</sup> storing each coalition, its prediction, and the corresponding kernel weight. 

### 3) GRAD-CAM (GRADIENT-WEIGHTED CLASS ACTIVATION MAPPING) 

For CNN-based models trained on spectrograms, GradCAM is used to generate visual explanations [18]. This model-specific technique produces a coarse heatmap highlighting the most salient regions influencing a class prediction. The procedural steps for generating a Grad-CAM heatmap are detailed in Algorithm 3. 

Grad-CAM involves two main steps. First, neuron importance weights _ωm_<sup>_t_for each feature map</sup><sup>_Fm_are computed via global</sup> average pooling over the gradients of the class score _s_<sup>_t_</sup> across spatial locations ( _p, q_ ), as defined in Equation (10). 



Here, _K_ is the total count of pixels within the feature map. Next, the heatmap highlighting class-relevant regions is obtained via a ReLU-applied weighted sum of feature maps: 



where _H_ CAM<sup>_t_denotes the Grad-CAM heatmap for class</sup><sup>_t_,</sup><sup>_ω_</sup> _m_<sup>_t_</sup> is the weight of feature map _F_<sup>_m_</sup> for class _t_ , and _F_<sup>_m_</sup> is the 

179743 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



**TABLE 5.** Model performance metrics across datasets using audio-only and multimodal inputs. 



_m_ -th feature map of the last convolutional layer. This produces a visual explanation highlighting the time-frequency regions in the spectrogram that most influence the model’s prediction. 

### **IV. RESULT ANALYSIS** 

### _A. QUANTITATIVE RESULTS_ 

This section presents the quantitative performance of the proposed deep learning architecture. To evaluate the model’s generalizability and effectiveness on both unimodal and multimodal inputs, the same core architecture was trained and tested for each experimental condition. Classification performance was measured using the standard metrics defined in Section III-D: Accuracy, Precision, Recall, and F1-score. Table 5 summarizes the key results, enabling direct comparison of the architecture’s efficacy across domains and input modalities. 

A primary observation from the results is the model’s exceptional performance on datasets characterized by clean, well-defined audio samples, such as SAS-KIIT, Audio MNIST, and CREMA-D, where accuracies consistently exceeded 99% in some cases. This validates the architecture’s fundamental capability for audio pattern recognition in controlled conditions. In contrast, the performance on the healthcare-focused datasets such as CIRCOR, ICBHI, and COUGHVID, while still robust, was more modest. This outcome is expected and highlights the significant challenges inherent in analyzing pathological and real-world biomedical audio, including patient-specific variability, background noise, and the subtle, often overlapping nature of pathological sound events. 

Investigation of input modalities showed that adding demographic data is context-dependent. For ICBHI, it consistently improved performance, indicating demographic information provides valuable complementary context for complex respiratory analysis. For CIRCOR and COUGHVID, no clear gain was observed, suggesting audio features alone were sufficiently discriminative. Comparison of MFCCs and spectrograms also showed task-dependent performance: spectrograms were slightly better for CIRCOR, while MFCCs achieved higher accuracy for ICBHI and CREMAD, highlighting the need for empirical evaluation when designing models. The multi-class nature of healthcare datasets like ICBHI adds complexity, lowering accuracy. 

To provide a more in-depth analysis of the model’s training behavior and classification performance, the training history, confusion matrices, and ROC curves were visualized. As presenting these detailed visualizations for every model across all six datasets would be impractical due to space constraints. Figure 11 shows a typical set of performance graphs for different models, illustrating the standard of convergence and evaluation achieved in this study. 

### _B. EXPLAINABLE AI (XAI) VISUALIZATIONS_ 

### 1) EXPLAINABILITY ON CIRCOR DATASET 

A comprehensive explainability analysis is presented for a representative heart sound from the CIRCOR dataset, in which the multimodal model accurately identified the sample as ‘Absent’ (i.e., no murmur detected), as illustrated in Figures 12 and 13. These visualizations, based on MFCC and spectrogram inputs, integrate local and global interpretability 

179744 

VOLUME 13, 2025 





<!-- Start of picture text -->
Multi-classi ROC Curve Confusion Matrix - Plasma Colorma) us<br>Afganisthan_Pashto_Music t t t 00<br>10. gioo a Bhuddist_PrayerAzan (‘ ‘‘ .‘<br>1 “¢ Children_class_noiseChurch_Prayer 96 | 4 (¢ ‘[ Pt<br>08 n o Dhak ( ( ¢<br>1 | iaoo” Fish_marketElephEkt a rant Elo «¢ ‘[‘ (c( 80<br>282@ 0-6 flCo 7ava ¢ 3g2E2 Irrigation_EngineHarmoniumFlute o¢ Ed 92 | (¢ 60<br>> yA Kalboishakhi_Storm ‘ ‘ c<br>B “ Launch_Engine o « 196] :<br>& o” Railway_Engine a ‘ ‘<br>o ie o” Rickshaw_Horn ( ( (<br>=> 04 | : a Sanatan_Religion_Aroti 1 ( r 4”<br>4Pua — Class Bronchiectasis (AUC = 0.93) Tabla o ¢ o<br>a —— Class Bronchiolitis (AUC = 0.92) Tiger ; ; ;<br>02. fili yyrea — Class COPD (AUCee = 0.96) Traditional_SongTanpura ti [ ”t ”<br>| ia — Class Healthy (AUC = 0.91) ZELSRSESESSELESLESESRE 20<br>o” —— Class Pneumonia (AUC= 0.91) Z*FESESCH SE SFR SSTAPSrA<br>—— Class URTI (AUC= 0.90) &@2g sss84 Be2 sgs3EgcsosssEee * g@B<br>0.00.0 &|s 8:6£83s EsS2552Zgs3pehRe5 3z<br>0.2 0.4 0.6 0.8 1.0 s & 3 ‘s 0<br>False Positive Rate 5 8<br>(a) 2 (b) Predicted label<br>Loss Curve Accuracy Curve Training and Validation Accuracy<br>3.0 —— Train Loss 1.0 08<br>— val Loss<br>2.5 08 0.7 |{<br>2.0<br>J<br>Z o6<br>Ls 06 S-<br><<br>1.0 os<br>0.4<br>0.5<br>02~<br>0.0 —— TrainVal Accuracy Accuracy 04 | — — TrainingValidation AccuracyAccuracy<br>° 20 40 60 80 100 0 20 40 60 80 100 0 10 20 30 40 50<br>Model Accuracy (c) Model Loss (d)<br>Lo 7s —— rain Loss Training and Validation Accuracy<br>—— Validation Loss<br>09 1.50 0.8<br>1.25<br>os 0.7<br>= 1.00<br>¥ 07 0.75 0.6<br>0.50<br>06 05<br>0.25<br>os — Tain Accuracy oe —— Training Accuracy<br>—— Validation Accuracy 0.00 —— Validation Accuracy<br>° 20 40 Epoch 60 80 100 (e) ° 20 40 Epoch 60 80 100 ° 5 10 15 20 25 30<br>Confusion Matrix Confusion Matrix 900<br>o BEER oO 3 ° ° ° ° 4 ° °<br>Disgust 994 14 13 25 3 o 14 o Bega o fe) fe) fe) ° fe) o 6 800<br>Angry 11 960 10 14 1 11 800 2 fe] fe] 899 °o fe] o o 1 o oO 700<br>34 0 ° o EeER o ° ° ° 1 ° 600<br>_ Fear 1 14 914 33 ES) 20 600<br>3B 2440 ° ° o EEeza ics o fe) [e) o 500<br>s 2<br>E Happy 7 15 12 954 11 1 200 =Ssio0 0 0 o o BR o 0 oo o 400<br>67 0 ° ° ° ° o Ee o ° ° 300<br>Neutral 1 33 3 28 814 13 200 7 ° ° ° °o ° ° o Eee o ° 200<br>Sad ° 37 24 zi 36 911 840 ° ° ° ° 2 fo} o Eee o 100<br>° 94 0 ° ° ° ° [e) ° (o) o Ei<br>B> => 2S raEs s3 3a ° y+ 7 > & oS ° A co 2 °<br>a =< = Zz Predicted label<br>Predicted label (h)<br><!-- End of picture text -->





<!-- Start of picture text -->
-<br>SST)=—__ LIME— Explanation2  fora '‘Absent'= (Top 10— Features) = —_—=x] NOT PresentHeight <= -0.6702 Present E:<br>ee“ BaeeS StPx; a?7 eeot Sreyel : ee ad |ee 3.2ee re we : Weightahex <= 0:soo S E<br>peHestmap of SHAR Cxplanations(a1)for Alt Samples | coarsezs”?7 - : Laoosoo<br>WeignenetoneAge ae Esp2 5 S za2 eae. a- - +: -o.5[wsas<br>sex = —2.0<br>preonancy status | = a i a 25<br>: , —_—., ic -o.oers —ie Ss oees reas<br>(b1) (b2)<br>ree Explanation (Prediction: Apes. Demographic Feature Importance (Prediction: Absent)<br>——_——— rs 1.325 = Height +0.03<br>1 sex 40.02<br>© = Pregnancy status +0.01<br>2.423= Weight —0.01<br>3 = Age 6<br>a —————— EE —o!01 0.00 0.01 0.02 0.03<br>— — — al = = = or value<br>Original MFCC i b3Derad-can Heatmap Superimposed Result bsb4Ds Contours<br>(Cc)<br><!-- End of picture text -->





<!-- Start of picture text -->
> rs LIME Explanation-Demographic Features<br>; ‘eomedeen folie status <= -0.28<br>| F pene Sere pabas ani Pregnancy<br>BAS ae stan ey 0.97 < sex <= 103<br>seeye 3/ Ly 0.06< Weight <= 0.35<br>. SihiB $ t 0.67 < Height <= 0.00<br>(a1) . fx) base value (a2) Contribution to prediction le-23<br>0.77 higher 2 lower<br>0.750 o.755 0.760 Demographics0.765 SHAP - 0770Explaining Prediction:0.775Absent (0) 0.780 0.785 0.790 0.795<br>a eeeee<br>Sex = 0.0 Age = 4.0 Weight = 0.8564000129699707 Height = 1.4318000078201294 Pregnancy status = 0.0<br>Absent (0) Present (1) (b1) Unknown (2) Global Feature Importance for Spectrograms<br>° (Explaining 'Present' Class)<br>Fd<br>so 0.0008 &<br>\ ‘ ~ 0.0006 &<br>{ Fa a<br>% 4 $s= 100 0.0004 32<br>' é _¥ = 150 2<br>er wwf 0.0002 §3<br>La : . , TT. 200 =<br>—0.002 —0.001 0.000 0.001 0.002 o 50 100 150 200<br>Shap value — (2) Time<br>Spectrogram with Grad-CAM Overlay (b3)<br>Decision Plot for All 5 Samples<br>0.74 0.76 0.78 0.80<br>Height<br>Weight<br>Age<br>Sex<br>:<br>0.0 o.1 0.2 Relevance 0.3 0.4 os Pregnancy status ET} oteModel output valueote ob<br>(c1) (b4)<br><!-- End of picture text -->





<!-- Start of picture text -->
High<br>NOT COPD COPD Age tt ee ee so see eee wees mfce_20<br>mfcc_20foc 0 > > -432 -432.230.06! ‘Adult_BMIsex ceceee eeete eeeee mfcc_21mfcc_o<br>~ 0.05 mfcc_355 .- mfcc_41,<br>mfec_ 21 <= 0.00 fee 201 ceeee mfce_<br>fcc 41<= 0.00 mfcc_353 toe mfcc_40<br>10.04oesfec_ — 1i <= 0.000. mfcc_352mfcc_416 eee— mfcc_280mfcc_60<br>mfcc_40>-438.81 mfcc_314 mee © mfcc_80<br>a cc_42 <= 0.00 mfcc_378 see  _ mcc_240<br>0.01 - mfcc_202 wee 2 mfcc_420<br>enfcc_666 > 24.75 mnfee 304 wee SB itce_a2<br>|-363.2510.01 < mfcc_380<~ mfcc_433mfcc_395 seeeee mfcc_480mfcc_580<br>358.26 <mfee 240<- yee a01 we mfec_500<br>Prediction probabilities mfcc_335 se mfcc_640<br>BronchiectasisPneumonia5 copD7 (EEE 0.09 mfcc_231mfcc_376 --La. mfcc_560fee 721 tems@@™m™mm COPDPneumoniaBronchiectasis<br>OtherURTI mfcc_354mfcc_494- ee~ mfcc_540mfcc_701- mmmmmmm URTIHealthyBronchiolitis<br>(a1) =0.0025 HAP-0.0020value (impact-0.0015 on-0.6010model output)0.0005 0.0000 =‘ ™ 0.00 mean(|SHAP0.02 value|) (average0.04 impact0.06on model output0.08magnitude)0.10<br>LIME Explanation for Class: COPD (b1) (b2)<br>mfcc_20 Grad-CAM Heatmap (Sample 5)<br> > -432.23 0.0 — ri 10<br>mfcc_0 > -432.94 25 | 4s i ; 0.9<br>mfcc_21 <= 0.00 5.0 | | 0.9<br>mfcc_41 <= 0.00 » | oes<br>mfcc_40mfcc_1- <= 0.00 €5g 75 08 gyEB NS oso7 2=&<br> > -438.81 g 70° 07 5 _ B<br>mifcc_666mfce_42 <=> 24.75 0.00 re)= 12.515.0 1 0.6 = ) 2 Ww~ 0?4005 O<br>-363.25 < mfcc_380 <= -297.39 i oo 20<br>17.5 F] E 05 25 so Bs<br>-358.26 < mfcc_240 <= -304.97 a Merce. 10.0 z 10<br>0.06 - 0 5 10 15 20 2 30 35 40 F 20.0<br>(a2)0.04 -0.02 0.00 002 0.04 “Time(c1) rrames (c2)><br><!-- End of picture text -->





<!-- Start of picture text -->
Side-by-Side Comparison for 101_1b1_Al_sc_Meditron.png<br>Original Spectrogram Grad-CAM Heatmap<br>* ba<br>£<br>rc. * —_ o4 E3E<br>, 0.0<br>(a1) (c1)<br>0.52 053 sa Demographics0.55 SHAP0.56 - Explainingf(x)057 057Prediction:oss0 ‘COPD’ 0.59 0.60 0.61 higher 7 lower 0.01925 0.01950SHAP0.01975Decision 0.02000Plot- Explaining0.02025‘Bronchiolitis*0.02050 0.02075 0.02100<br>|eee ee AgeSex (0.148) qa)<br>Adult_BMI (0.261)<br>Age = 0.6010781526565552|Sex = 0.0 Adult_BMI = 0.22239629924297333 0.01925 0.01950 0.01975 Model0.02000output0.02025value 0.02050 0.02075 0.02100<br>(b1) (b2)<br>Pneumonia Bronchiectasis Bronchiolitis coPD Healthy Pneumonia URTI<br>*P 4 Pen): 3;<br>: ii “<br>0.004 0.003 -0.002 0.001 0.000 0.001 0.002 0.003 0.004<br>SHAP value<br>(b3)<br><!-- End of picture text -->





<!-- Start of picture text -->
--- Demographic Data ---<br>Age: 15<br>Gender: Female<br>Respiratory Condition: No<br>Fever/Muscle Pain: No<br>--- Model Prediction ---<br>True Label: Healthy<br>Predicted Label: Healthy<br>Confidence: 86.85%<br>(a1) fix) = 0.415<br>~ lage 30.00 -_<br>25.00 < age <= 35.00 respiratory_condition 1.00<br>respiratory_condition >...a fever_muscle3 -_paini 0.00<br>10.00 < gender <= 1.00 Prediction probabilities 0278 ~ Feature 2<br>00 healthy NNd.87 0.532 - Feature<br>fever_muscle_pain <=...ou symptomaticCOVID-19 —0.283 = Feature o4 |+0| +0<br>Grad-CAMm: Model Importance(a2)per Frequency Bin; 0.37a0.38 0.39 0.40 FLAX] =0.416.405 0.42<br>° (b1)<br>> 3<br>10° \<br>= os<br>& on 8<br>= 20 o2 2<br>= oo &<br>30 5.07 2s 30<br>3s “a 10.0 20<<br>“ee aes 15.0 8 oo<br>a0 oo oe Totalos Importance (Attention)oe os ze a7 20.0 ° .<br>(c1) (c2)<br><!-- End of picture text -->



<!-- Start of picture text -->
oat 1 2 J GANSwe aes eel a “<br>: _ a ae =— = = =<br>=oS—_ ia = = —rie aere >  —_— a—<br>COvID-19 (a1) Healthy Symptomatic (a2) coviD-19<br>RAL agit be ¥; ie tel Vee<br>t . e<br>—o.0002 —o.0001 0.0000 e.0001A  _ 0.0002<br>SHAP value<br>° Grad-CAM Contours on Spectrogram (b1) ° Grad-CAM Heatmap for Spectrogram (Predicting ‘Healthy')<br>20 BF 2 = F oS > 20 oe<br>2 op Zz < a 20<br>3 aS ~\ aL Fa =<br>E s0 eS hee oe< SE5N\)= £ 00 oa E<br>120 oe et2 Sa na SSa -> Re ao oz<br>° 20 40 sire,6 prames 80 100 120 6 20 4 Time‘co Frames 30 100 320<br>(c1) (c2)<br><!-- End of picture text -->





<!-- Start of picture text -->
arr ; 7<br>—._. | fw | ong<br>ber iar’ bP F<br>em- 5 = ee s \ ¢ “. 1 a a 038s<br>™ o2€<br>= 422m, ae<br>oo<br>Cal) Ca2)<br>Original MFCC SHAP Values Grad-CAM: Model Importance per Frequency Bin<br>g3 10 is y yan aanI 7 be pyade ibaaaaa rae DevePoe oo wn egiesCapita ceaateWeaknee cetlle me,os<br>9 30 f ‘ a yak eaateeys iretae atta ee orbsting saint Becta ie<br>5 . es if + de US SOR i ee Sea ea eetRe ta,<br>° 25 50 75 100 125 150 175 ° 25 50 75 100 125 150 175<br>Time Steps Time Steps < 40<br>Le ><br>-0.4 -0.3 -0.2 -0.1 0.0 01 0.2 0.3 04 3<br>SHAP Value (impact on Model Output) © oo<br>cb) z<br>° Grad-CAM Heatmap on MFCC =<br>8<br>5 O08 reyZ 80<br>2 10 =<br>5 1s 06 y<br>€<br>3 20 £ 100<br>6 04s<br>3 75 E<br>= 30 02<br>35 120<br>40 0.0<br>0.0 os 10 Ls 2.0 2-5 3.0 3.5 4.0 0.0 O2 0.4 0.6 os Lo<br>Time (seconds) Total Importance (Attention)<br>cc) ce2)<br><!-- End of picture text -->



<!-- Start of picture text -->
—=— OS,eg — _—— — —— : ’ = = — = =<br>a rr rt ee ee ee — eet<br>° (a1) a SHAP Importance Heatmap(a2)<br>EaBass= 1007s ¥ 4:: eccceo.cco. S==B (Ss2—#ze] p |HisFe Bgjsspag5 SSR e eS eSe ESEEaeSSSSSSSeeBSSESEHS= ecco_c.cozc=} = =<br>asoi “6 ¢SER Si, ‘) o.cce2 S Hatt Pia - : == =$ ——— -c.cos 2&<br>200 i=eeTapeerewecy a= Pre < ERSeesRe eee<br>Ss SS eS ab ES ISO ES SOO _o.00e<br>(b1) (b2)<br>ooo —Grad-CAM Contours—————on Spectrogram — _[-———_—-° Global Attention Map (Averaged Grad-cAm)<br>S2”co ——= S2”co : —_": e208g<br>E 20 Eso ons<br>2sooaa = —— rooeee one—<br>os os =o wee == =o = i bs =o a sso aio<br>(e1) (c2)<br><!-- End of picture text -->





<!-- Start of picture text -->
High LIME Explanation as Heatmap<br>Faa H — — ————_= = .<br>F_1603Fas H i — ——————= _ = om = ' —<_ —= =2oe oo i 03 _S=<br>- — <_< a =<br>F.1643 I —— oS 8<br>F_1523F_1663 { i ' = oh& —_ ' =——-=.—_:i.SS a. =.: == —-—os1. ol =EaS<br>== == = rT... oo 62<br>Faze1 I ; a 2<br>F_1003 ai — = ‘hs s<br>i » : . “a = <&<br>F_902 ! 3 = a —0.1<br>-F_34 i s25 BE = _ —= (a1 )<br>F_1540 { Grad-CAM: Model Importance per Frequency Bin<br>F885 i Oth F_O +4 0.0<br>F185~ | Ist F_O +0 E25.<br>F_1002 H £ 50<br>F_1300 i 2nd F_O *0 gE 7<br>F_1541~ i 4th F_O +0 €s 10.0<br>F_1521F_1560 iH 3rd F_O +0 §8g 12.515.0<br>F_1683 i 2azs<br>5th FO +0<br>—0.010 —0.005 0.000 0.005 0.010 Low 7 20.00.0 0.2 0.4 0.6 08 10<br>SHAP value (impact on model output) 0.00000 0.00005 0.00010 0.00015 0.00020 0.00025 0.00030 0.00035 0.00040 Total Importance (Aggregated Attention)<br>(b1) (b2) (c1)<br><!-- End of picture text -->



<!-- Start of picture text -->
Positive Contributions (supports prediction) Negative Contributions (opposes prediction) Average SHAP Heatmap over 10 Samples — ay |<br>°asg&Eda<br>——_——————s<br>(at) (b1)<br>ities i<br>- —<br>° —0.0004 —0.6002 SHAP0.0000value (B2) 0.0002 3D Grad-CAmM 0.0004<br>=o 1°<br>20 eee x os<br>= so AAX SD 2<br>soo < 12°<br>20 = zo ne ee oewoe<br>°.°0 o2 ‘Total tmportanceo.4 (Aggregatedo.6 Attention)os 1.90 ated Fame, 2 19° 220 ° =e<br>(c1) (c2)<br><!-- End of picture text -->





<!-- Start of picture text -->
True Label: 6 | Model's Top Prediction: 6<br>° 1.100 Feature 303 Le<br>Zs2= 1.025 =Feature8 Feature 306363 | ESee<br>=&3Es7° 097551.000 PrE& FeatureFeature 366353 eee|<br>20 0.950 Feature 497 ee<br>Feature 356<br>35 0.925 Feature 626 |<br>° 20 40 Time60 Frames 80(at) 100 120 0.900 0.0002(b1) 0.0001‘SHAP value (impact(0.0000on mode! output)(0.0001 (0.0002<br>= uy * a 1e—7 2 Original_ MFCC 7 Grad-CAM Heatmap Superimposed. Result Lo<br>- - 5 wia's . = oy 2 naj lnegea-etous bt-aeecke feeep tp algae<br>=a 5 js 2 romain . 08§<br>= aig==——oe heahs ns . . °a23a=g #\ I i1 | 06 §223gH<br>.=a =| =,bai-2 »oan3<br>=‘_ — ge E<br>= = es | 3 022<br>Looe —(b2)a enen as -4 (c1) 00<br><!-- End of picture text -->



<!-- Start of picture text -->
True: 0/ Pred: 0 True: 1/ Pred: 1 True: 2/ Pred: 2 True: 3 / Pred: 3 True: 4/ Pred: a<br>> i = 5 L<br>: ed=~_ :=>> f , q. =z—— c|J=3- = —<br>: =. — Ba — > CS =<br>— ce = = ‘ _ =. — f tS aie<br>rue: 5 / Pred: 5 True: 6 / Pred: 6 True: 7 / Pred: 7 True: 5 / Pred: 5 True: 9 / Pred: 9<br>=j 5 L } =— a f=»i } = ,<br><qxCef —_=: |2 Q=|tu 3 Es5 | se =,Q~~ 5 | c |1t c——..3 i t a———ta i —3 —. uw3 —— 4<br>3 ee= = =os ly 2 ? — a ; 7 : efiF i > ay= fe a2. - <=2 cyA Be i<br>7 7 o (at) 3 6<br>——=<br>=-=_—=_<br>-——id= -= =_ = =<br>—0.0002 —o.0001 0.0000 0.0001 0.0002<br>SHAP value<br>_S 10 Grad-CAM: Model Attention Over Time (b1) 20°o Grad-CAM as an Importance Mask<br>Eos<br>£ 2 40<br>=g2os0=o2>°°<br>£s<br>£g0.4 &= 80 == SS<br>B02 200 ———<br>0.0 oO 20 40 60 80 100 120 0.0 os 1.0 1s 2.0 2.5 3.0 3.5 4.0<br>Time Frames Time (seconds)<br>(c1) (c2)<br><!-- End of picture text -->

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



**TABLE 6.** Comparative evaluation of XAI methods across multiple datasets and input representations, measured in terms of fidelity, stability, and sparsity. 



input features: 



Here, | _Hi_ | is the number of highlighted features for sample _i_ , and _F_ tot is the total number of input features (MFCC coefficients or spectrogram bins). Since audio clips vary in length, _F_ tot also varies across samples, which naturally introduces differences in sparsity values. 

### 2) UNIFIED QUANTITATIVE RESULTS 

Table 6 presents the fidelity, stability, and sparsity for LIME, SHAP, and Grad-CAM across all datasets and input modalities. SHAP consistently achieves the highest fidelity, indicating robust identification of influential features. Grad-CAM applied to MFCC inputs effectively highlights time–frequency regions, validating its methodological soundness. Incorporating demographic features slightly improves fidelity without affecting sparsity, confirming the complementarity of multimodal inputs. Sparsity values remain low overall, reflecting concise explanations; their variation is influenced by the total number of input features ( _F_ tot), which depends on the audio length and feature extraction method. Stability remains consistently high, demonstrating the reproducibility of the explanations across runs. 

### 3) JUSTIFICATION FOR GRAD-CAM ON MFCC FEATURES 

MFCCs are represented as _2D arrays_ (time × coefficient), enabling convolutional neural networks to capture local temporal–spectral patterns. Grad-CAM applied to CNN feature maps produces attention maps highlighting influential time–frequency regions. This approach, consistent with prior audio AI studies, shows that Grad-CAM on MFCCs provides interpretable and meaningful insights. 

### 4) NOTES ON METRICS 

- Fidelity and stability values lie in [0,1], with higher values indicating better alignment with model predictions and greater reproducibility. 

- Sparsity is normalized to [0,1]; lower values indicate more concise explanations highlighting fewer features. 

- Insertion AUC is reported as fidelity; deletion AUC was also evaluated with consistent trends but omitted for brevity. 

### **V. DISCUSSION** 

This study conducted a cross-domain evaluation of a unified deep learning architecture for audio classification, with a central focus on analyzing model behavior through Explainable AI (XAI). By combining model performance outcomes (Table 5) with a quantitative evaluation of explanations 

179754 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



**TABLE 7.** Summary of key explainability findings across all datasets. 



(Table 6), we critically assess the reliability, complementarity, and practical implications of XAI in diverse audio intelligence tasks. The model’s legitimacy is first established by predictive accuracy. On benchmark datasets like CREMA-D, Audio MNIST, and SAS-KIIT, performance was consistently strong (>95%), validating the CNN’s ability to capture robust acoustic patterns. More modest but stable performance on clinical datasets (CIRCOR, ICBHI, CoughVID) underscores the challenges of real-world medical audio, including noise, inter-patient variability, and subtle pathological cues. 

Beyond classification accuracy, the main contribution lies in systematically assessing explanations. The three XAI 

methods, SHAP, LIME, and Grad-CAM, offer complementary perspectives, as summarized in Table 7, which presents a concise overview of the key findings. Numerical comparisons (Table 6) show that SHAP consistently achieved the highest fidelity (e.g., 0 _._ 85 ± 0 _._ 02 on ICBHI multimodal vs. 0 _._ 83 ± 0 _._ 02 for LIME and 0 _._ 81 ± 0 _._ 03 for GradCAM) and the lowest sparsity, producing concise and stable explanations. LIME and Grad-CAM, while less precise, add interpretive richness and computational efficiency. A key insight from cross-domain analysis is that while the specific acoustic features highlighted by explanations are domain-dependent, such as prosodic cues in speech 

179755 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



emotion recognition or systolic/diastolic segments in heart sounds, the interpretive role of XAI generalizes across contexts. In every task, explanations consistently surfaced the most task-relevant features, demonstrating that the framework is transferable even when the underlying signals vary. 

Multimodal experiments further revealed how demographic variables interact with acoustic inputs. Explanations attributed predictions to both audio features and metadata (e.g., age, BMI), enabling richer and more patientspecific interpretations. Importantly, no single demographic factor disproportionately dominated attributions, supporting fairness in model behavior. Another practical consideration concerns computational feasibility. Perturbation-based methods such as LIME and SHAP are computationally demanding, making them best suited for offline auditing and validation. In contrast, Grad-CAM is lightweight and thus more viable for real-time or embedded systems. This distinction enables a division of labor: Grad-CAM for rapid deployment, SHAP for rigorous validation, and LIME as an accessible middle ground for interpretability checks. 

Taken together, these findings motivate several practical guidelines. For healthcare applications, SHAP emerges as the preferred method due to its strong fidelity, stability, and theoretical grounding. For real-time or embedded systems, Grad-CAM offers the most feasible solution given its computational efficiency. In the context of general debugging and model development, a hybrid strategy is recommended: Grad-CAM provides quick focus visualizations, LIME enables intuitive local checks, and SHAP supplies robust global attributions. Finally, the choice of feature representation should also be task-driven: spectrograms are highly effective when human-centric interpretability is the priority, as Grad-CAM heatmaps align naturally with their time–frequency axes, while MFCCs are more appropriate when predictive accuracy and compactness are paramount. 

Finally, we acknowledge a limitation regarding the inclusion of demographic data in our multimodal models. While feature attribution analyses suggest that predictions were not overly influenced by demographic factors, a formal quantitative fairness assessment such as evaluating demographic parity or other established metrics was not conducted. Conducting a rigorous fairness evaluation remains an important direction for future work. We note a dimensionality mismatch between MFCCs (40-D) and spectrograms (128×128), which may influence interpretability outcomes. While this choice follows common practice, it does not fully disentangle feature type from dimensionality. Matching feature dimensions or formally studying its impact on interpretability remains a potential avenue for future work. 

Together, these insights offer a practical roadmap for deploying XAI in audio intelligence, ensuring transparency, robustness, and domain relevance across both clinical and non-clinical contexts. 

### **VI. CONCLUSION AND FUTURE WORK** 

This paper presented a comprehensive cross-domain analysis evaluating three prominent XAI techniques, LIME, SHAP, and Grad-CAM in diverse audio intelligence tasks. Moving beyond purely qualitative assessments, we introduced a rigorous quantitative framework to measure the quality of explanations. Our evaluation, based on fidelity, stability, and sparsity metrics, demonstrated that while all tested methods produce reliable explanations, SHAP consistently offers the highest fidelity and the most concise outputs. Furthermore, we identified a key practical trade-off for practitioners: spectrograms generally yield more intuitive explanations for human experts, whereas MFCCs can provide superior predictive accuracy in certain tasks. These findings, distilled into a set of practical guidelines, contribute to a deeper and more nuanced understanding of how to build transparent and trustworthy audio AI systems. Building on this work, several important avenues for future research are apparent. This framework could be extended to other advanced architectures, such as attention-based Transformers, which may yield different explainability patterns. A crucial next step is to conduct a formal quantitative fairness audit, expanding on the preliminary investigation presented here to ensure models are equitable across demographic subgroups. Other promising directions include developing interactive explanation interfaces for real-time clinical use and pursuing formal validation of these explanations with domain experts to confirm their practical utility. 

### **REFERENCES** 

- [1] V. Bento, M. Kohler, P. Diaz, L. Mendoza, and M. A. Pacheco, ‘‘Improving deep learning performance by using explainable artificial intelligence (XAI) approaches,’’ _Discover Artif. Intell._ , vol. 1, no. 1, p. 9, Dec. 2021. 

- [2] M. Swain, A. Routray, and P. Kabisatpathy, ‘‘Databases, features and classifiers for speech emotion recognition: A review,’’ _Int. J. Speech Technol._ , vol. 21, no. 1, pp. 93–120, Mar. 2018. 

- [3] K. Zaman, M. Sah, C. Direkoglu, and M. Unoki, ‘‘A survey of audio classification using deep learning,’’ _IEEE Access_ , vol. 11, pp. 106620–106649, 2023. 

- [4] M. French and R. Handy, ‘‘Spectrograms: Turning signals into pictures,’’ _J. Eng. Technol._ , vol. 24, no. 1, p. 32, 2007. 

- [5] Z. K. Abdul and A. K. Al-Talabani, ‘‘Mel frequency cepstral coefficient and its applications: A review,’’ _IEEE Access_ , vol. 10, pp. 122136–122158, 2022. 

- [6] G. Adomavičius and A. Tuzhilin, ‘‘Expert-driven validation of rule-based user models in personalization application,’’ _Data Mining Knowl. Discovery_ , vol. 5, pp. 33–58, Oct. 2001. 

- [7] E. LaRosa and D. Danks, ‘‘Impacts on trust of healthcare AI,’’ in _Proc. AAAI/ACM Conf. AI, Ethics, Soc._ , Dec. 2018, pp. 210–215. 

- [8] R. Chatterjee, P. Bishwas, S. Chakrabarty, and T. Bandyopadhyay, ‘‘South Asian sounds: Audio classification,’’ in _Proc. 4th Int. Conf. Comput., Commun., Control Inf. Technol. (C3IT)_ , Sep. 2024, pp. 1–6. 

- [9] D. Gunning, ‘‘Explainable artificial intelligence (XAI),’’ _Defense Adv. Res. Projects Agency (DARPA)_ , vol. 2, no. 2, pp. 1–11, 2017. 

- [10] M. Ribeiro, S. Singh, and C. Guestrin, ‘‘‘Why should i trust you?’: Explaining the predictions of any classifier,’’ in _Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics, Demonstrations_ , 2016, pp. 1135–1144. 

- [11] S. Lundberg and S. Lee, ‘‘A unified approach to interpreting model predictions,’’ in _Proc. Adv. Neural Inf. Process. Syst._ , 2017, pp. 1–8. 

- [12] D. Vij, Y. Yogesh, D. Srivastava, and H. Shankar, ‘‘Detection of acoustic scenes and events using audio analysis—A survey,’’ in _Proc. 3rd Int. Conf. Advance Comput. Innov. Technol. Eng. (ICACITE)_ , May 2023, pp. 316–320. 

179756 

VOLUME 13, 2025 

S. Chakrabarty et al.: Can We Trust AI With Our Ears? A Cross-Domain Comparative Analysis 



- [13] N. K. Chowdhury, M. A. Kabir, M. M. Rahman, and S. M. S. Islam, ‘‘Machine learning for detecting COVID-19 from cough sounds: An ensemble-based MCDM method,’’ _Comput. Biol. Med._ , vol. 145, Jun. 2022, Art. no. 105405. [Online]. Available: https://www. sciencedirect.com/science/article/pii/S0010482522001974 

- [14] D. Stowell, D. Giannoulis, E. Benetos, M. Lagrange, and M. D. Plumbley, ‘‘Detection and classification of acoustic scenes and events,’’ _IEEE Trans. Multimedia_ , vol. 17, no. 10, pp. 1733–1746, Oct. 2015. 

- [15] W. C. Sleeman, R. Kapoor, and P. Ghosh, ‘‘Multimodal classification: Current landscape, taxonomy and future directions,’’ _ACM Comput. Surveys_ , vol. 55, no. 7, pp. 1–31, Dec. 2022, doi: 10.1145/3543848. 

- [16] R. Dwivedi, D. Dave, H. Naik, S. Singhal, R. Omer, P. Patel, B. Qian, Z. Wen, T. Shah, G. Morgan, and R. Ranjan, ‘‘Explainable AI (XAI): Core ideas, techniques, and solutions,’’ _ACM Comput. Surveys_ , vol. 55, no. 9, pp. 1–33, Jan. 2023, doi: 10.1145/3561048. 

- [17] J. Van Der Waa, E. Nieuwburg, A. Cremers, and M. Neerincx, ‘‘Evaluating XAI: A comparison of rule-based and example-based explanations,’’ _Artif. Intell._ , vol. 291, Feb. 2021, Art. no. 103404. 

- [18] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ‘‘Grad-CAM: Visual explanations from deep networks via gradient-based localization,’’ in _Proc. IEEE Int. Conf. Comput. Vis. (ICCV)_ , Oct. 2017, pp. 618–626. 

- [19] Y. LeCun, Y. Bengio, and G. Hinton, ‘‘Deep learning,’’ _Nature_ , vol. 521, no. 7553, pp. 436–444, 2015. 

- [20] E. Glikson and A. W. Woolley, ‘‘Human trust in artificial intelligence: Review of empirical research,’’ _Acad. Manage. Ann._ , vol. 14, no. 2, pp. 627–660, Jul. 2020. 

- [21] W. Samek, T. Wiegand, and K.-R. Müller, ‘‘Explainable artificial intelligence: Understanding, visualizing and interpreting deep learning models,’’ 2017, _arXiv:1708.08296_ . 

- [22] F. Doshi-Velez and B. Kim, ‘‘Towards a rigorous science of interpretable machine learning,’’ 2017, _arXiv:1702.08608_ . 

- [23] E. Tjoa and C. Guan, ‘‘A survey on explainable artificial intelligence (XAI): Toward medical XAI,’’ _IEEE Trans. Neural Netw. Learn. Syst._ , vol. 32, no. 11, pp. 4793–4813, Nov. 2021. 

- [24] A. Holzinger, C. Biemann, C. S. Pattichis, and D. B. Kell, ‘‘What do we need to build explainable AI systems for the medical domain?’’ 2017, _arXiv:1712.09923_ . 

- [25] A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, and F. Herrera, ‘‘Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,’’ _Inf. Fusion_ , vol. 58, pp. 82–115, Jun. 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1566253519308103 

- [26] A. Akman and B. W. Schuller, ‘‘Audio explainable artificial intelligence: A review,’’ _Intell. Comput._ , vol. 3, p. 74, Jan. 2024. 

- [27] C. O. Retzlaff, A. Angerschmid, A. Saranti, D. Schneeberger, R. Röttger, H. Müller, and A. Holzinger, ‘‘Post-hoc vs ante-hoc explanations: XAI design guidelines for data scientists,’’ _Cognit. Syst. Res._ , vol. 86, Aug. 2024, Art. no. 101243. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1389041724000378 

- [28] N. Sobahi, O. Atila, E. Deniz, A. Sengur, and U. R. Acharya, ‘‘Explainable COVID-19 detection using fractal dimension and vision transformer with grad-CAM on cough sounds,’’ _Biocybernetics Biomed. Eng._ , vol. 42, no. 3, pp. 1066–1080, Jul. 2022. 

- [29] L. Orlandic, T. Teijeiro, and D. Atienza, ‘‘The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms,’’ _Sci. Data_ , vol. 8, no. 1, p. 156, Jun. 2021, doi: 10.1038/s41597-021-009374. 

- [30] G. Chaudhari, X. Jiang, A. Fakhry, A. Han, J. Xiao, S. Shen, and A. Khanzada, ‘‘Virufy: Global applicability of crowdsourced and clinical datasets for AI detection of COVID-19 from cough,’’ 2020, _arXiv:2011.13320_ . 

- [31] D. Bhattacharya, N. K. Sharma, D. Dutta, S. R. Chetupalli, P. Mote, S. Ganapathy, C. Chandrakiran, S. Nori, K. K. Suhail, S. Gonuguntla, and M. Alagesan, ‘‘Coswara: A respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection,’’ _Sci. Data_ , vol. 10, no. 1, p. 397, Jun. 2023. 

- [32] J. S. Ramakrishna, S. C. Venkateswarlu, K. N. Kumar, and P. Shreya, ‘‘Development of explainable machine intelligence models for heart sound abnormality detection,’’ _Indonesian J. Electr. Eng. Comput. Sci._ , vol. 36, no. 2, p. 846, Nov. 2024. 

- [33] E. Mahamud, N. Fahad, M. Assaduzzaman, S. M. Zain, K. O. M. Goh, and M. K. Morol, ‘‘An explainable artificial intelligence model for multiple lung diseases classification from chest X-ray images using fine-tuned transfer learning,’’ _Decis. Analytics J._ , vol. 12, Sep. 2024, Art. no. 100499. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S2772662224001036 

- [34] A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, ‘‘Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks,’’ in _Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)_ , Mar. 2018, pp. 839–847. 

- [35] N. T. Pham, S. D. Nguyen, V. S. T. Nguyen, B. N. H. Pham, and D. N. M. Dang, ‘‘Speech emotion recognition using overlapping sliding window and Shapley additive explainable deep neural network,’’ _J. Inf. Telecommun._ , vol. 7, no. 3, pp. 317–335, Jul. 2023. 

- [36] F. S. Prity, M. M. Hossain, M. M. Hossain, M. S. Uddin, M. R. Islam, M. Raquib, M. R. Ali, and K. M. A. Uddin, ‘‘Artificial intelligencepowered environmental sound recognition with explainable AI techniques,’’ _Iran J. Comput. Sci._ , vol. 2025, pp. 1–34, Jun. 2025. 

- [37] M. Shen, P. Mortezaagha, and A. Rahgozar, ‘‘Explainable artificial intelligence to diagnose early Parkinson’s disease via voice analysis,’’ _Sci. Rep._ , vol. 15, no. 1, p. 11687, Apr. 2025. 

- [38] F. Prior, T. Virmani, A. Iyer, L. Larson-Prior, A. Kemp, Y. Rahmatallah, L. Pillai, and A. Glover. (Aug. 2023). _Voice Samples for Patients With Parkinson’s Disease and Healthy Controls_ . [Online]. Available: https:// figshare.com/articles/dataset/VoiceSamplesforPatientswithParkinsons DiseaseandHealthyControls/23849127 

- [39] L. Tang, Z. Zhang, F. Feng, L.-Z. Yang, and H. Li, ‘‘Explainable Alzheimer’s disease detection using linguistic features from automatic speech recognition,’’ _Dementia Geriatric Cognit. Disorders_ , vol. 52, no. 4, pp. 240–248, 2023. 

- [40] A. A. Alyoubi and B. A. Alyoubi, ‘‘Interpretable multimodal emotion recognition using optimized transformer model with SHAP-based transparency,’’ _J. Supercomput._ , vol. 81, no. 9, p. 1044, Jun. 2025. 

- [41] S. Wang, J. Hu, Y. Du, X. Yuan, Z. Xie, and P. Liang, ‘‘WCFormer: An interpretable deep learning framework for heart sound signal analysis and automated diagnosis of cardiovascular diseases,’’ _Expert Syst. Appl._ , vol. 276, Jun. 2025, Art. no. 127238. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0957417425008607 

- [42] N. Das, N. Padhy, N. Dey, H. Paul, and S. Chowdhury, ‘‘Exploring explainable AI methods for bird sound-based species recognition systems,’’ _Multimedia Tools Appl._ , vol. 83, no. 24, pp. 64223–64253, Jan. 2024. 

- [43] S. Becker, J. Vielhaben, M. Ackermann, K.-R. Müller, S. Lapuschkin, and W. Samek, ‘‘AudioMNIST: Exploring explainable artificial intelligence for audio analysis on a simple benchmark,’’ _J. Franklin Inst._ , vol. 361, no. 1, pp. 418–428, Jan. 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0016003223007536 

- [44] S. Sultana, A. B. M. A. Hossain, and J. Alam, ‘‘COVID-19 detection from optimized features of breathing audio signals using explainable ensemble machine learning,’’ _Results Control Optim._ , vol. 18, Mar. 2025, Art. no. 100538. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2666720725000244 

- [45] M. Norval and Z. Wang, ‘‘Explainable artificial intelligence techniques for speech emotion recognition: A focus on XAI models,’’ _Inteligencia Artif._ , vol. 28, no. 76, pp. 85–123, Jun. 2025. 

- [46] Y. Choi and H. Lee, ‘‘Interpretation of lung disease classification with light attention connected module,’’ _Biomed. Signal Process. Control_ , vol. 84, Jul. 2023, Art. no. 104695. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1746809423001283 

- [47] T.-W. Kim and K.-C. Kwak, ‘‘Speech emotion recognition using deep learning transfer models and explainable techniques,’’ _Appl. Sci._ , vol. 14, no. 4, p. 1553, Feb. 2024. 

- [48] S. Shokouhmand, M. M. Rahman, M. Faezipour, and S. Bhatt, ‘‘Adventitious pulmonary sound detection: Leveraging SHAP explanations and gradient boosting insights,’’ in _Proc. 46th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)_ , Jul. 2024, pp. 1–4. 

- [49] I. D. Mienye, G. Obaido, N. Jere, E. Mienye, K. Aruleba, I. D. Emmanuel, and B. Ogbuokiri, ‘‘A survey of explainable artificial intelligence in healthcare: Concepts, applications, and challenges,’’ _Informat. Med. Unlocked_ , vol. 51, Mar. 2024, Art. no. 101587. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2352914824001448 

- [50] P. Seth and V. K. Sankarapu, ‘‘Bridging the gap in XAI-why reliable metrics matter for explainability and compliance,’’ 2025, _arXiv:2502.04695_ . 

179757 

VOLUME 13, 2025 

IEEE Access 



<!-- Start of picture text -->
uke<br><!-- End of picture text -->








# **Sci-Phi: A Large Language Model Spatial Audio Descriptor** 

## **Xilin Jiang**<sup>**1**</sup> **, Student Member, IEEE, Hannes Gamper**<sup>**2**</sup> **, Member, IEEE and Sebastian Braun**<sup>**2**</sup> **, Senior Member, IEEE** 

1Columbia University, New York, NY, USA 

2Microsoft Research, Redmond, WA, USA 

Work completed during internship at Microsoft. 

**ABSTRACT** Acoustic scene perception involves describing the type of sounds, their timing, their direction and distance, as well as their loudness and reverberation. While audio language models excel in sound recognition, single-channel input fundamentally limits spatial understanding. This work presents _Sci-Phi_ , a spatial audio large language model with dual spatial and spectral encoders that estimates a complete parameter set for all sound sources and the surrounding environment. Learning from over 4,000 hours of synthetic first-order Ambisonics recordings including metadata, _Sci-Phi_ enumerates and describes up to four directional sound sources in one pass, alongside non-directional background sounds and room characteristics. We evaluate the model with a permutation-invariant protocol and 15 metrics covering content, location, timing, loudness, and reverberation, and analyze its robustness across source counts, signal-to-noise ratios, reverberation levels, and challenging mixtures of acoustically, spatially, or temporally similar sources. Notably, _Sci-Phi_ generalizes to real room impulse responses with only minor performance degradation. Overall, this work establishes the first audio LLM capable of full spatial-scene description, with strong potential for real-world deployment. Demo: https://sci-phi-audio.github.io/demo 

**INDEX TERMS** Spatial audio, large language model, acoustic scene understanding. 

### **I. Introduction** 

A spatial acoustic scene is an organic whole of multiple sound events and ambient noise, together with the environment that shapes them. It includes source identity and content; onsets, offsets, and overlaps; direction and distance; loudness and reverberation; and the room’s overall imprint. These aspects are intertwined, and human listeners naturally bind them into a stable, unified representation: psychophysics studies on human hearing [1], [2] have shown that perception groups soundscape into coherent auditory objects and scenes using patterns over time and space, with distance and reverberation shaping where sources seem to be and spatial structure enabling selective listening in clutter. To fully analyze and understand an acoustic scene, one needs to detect multiple sources alongside background, track them along time, localize them in azimuth and elevation, estimate distance and level, and characterize the room. 

To solve this task, neural network-based machine listeners have progressed along several strands: sound event detection and localization [3], [4]; automatic speech recognition (ASR) [5], [6]; and general-purpose audio understanding with emergent audio large language models (LLMs) [7], 

[8]. Yet despite strong task performance, these models still fall short of perceiving an acoustic scene as an integrated whole: they typically focus on a single (or dominant) foreground source, omit spatial parameters (e.g., direction and distance), and offer little account of the environment (e.g., reverberation, room volume, noise). This gap motivates us to generalize machine listening from recognizing single auditory objects to narrating entire acoustic scenes. Our research goal is therefore twofold: (i) to investigate whether a machine can understand the entire spatial acoustic scene, including _what_ , _when_ , _where_ , and _how_ of the sound sources, and the _environment_ , analogous to human perception; and (ii) to build a spatial audio understanding model that can be extended to downstream applications including hearing assistants, robotics perception, navigation, and automatic spatial environment monitoring and annotation. 

This work introduces **_Sci-Phi_** , _<u>Spatial-scene comprehension</u> and_ _<u>inference</u> with Phi_ , the first spatial audio LLM capable of full spatial-scene description. _Sci-Phi_ builds on Phi-4 Multimodal [12], a powerful multimodal LLM for audio understanding and speech recognition that is nevertheless restricted to single-channel 

This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/ 

VOLUME , 

1 

Jiang et al.: 



<!-- Start of picture text -->
Model Format Localization Speech Audio Noise Acoustics Test on<br>Params Real RIR<br>BAT [9] Binaural 3D + dist. ✗ ✓ ✗ ✗ ✗<br>Can LLM...? [10] FOA 2D angle en ✗ ✗ ✗ ✗<br>SING [11] Owlet (1ch) 2D angle en ✗ ✗ ✗ ✗<br>Phi-4 MM [12] Monaural ✗ 8 lang. ✓ ✗ ✗ n.a.<br>Sci-Phi FOA 3D + dist. 8 lang. ✓ Type, Loud- Loudness, ✓<br>ness reverb,<br>room size<br><!-- End of picture text -->



<!-- Start of picture text -->
Large Language Model   Phi-4-Mini          Spatial LoRA<br><!-- End of picture text -->



<!-- Start of picture text -->
Audio LoRA<br><!-- End of picture text -->



<!-- Start of picture text -->
Spatial Projector<br><!-- End of picture text -->



<!-- Start of picture text -->
Text Encoder<br><!-- End of picture text -->



<!-- Start of picture text -->
Audio Encoder<br><!-- End of picture text -->

**TABLE 1. Existing (spatial) audio LLMs vs.** **_Sci-Phi_ , highlighting** **_Sci-Phi_ ’s full scene description and generalization ability.** 



audio input. At a high level, _Sci-Phi_ couples a _spatial encoder_ with an _audio encoder_ and is trained to generate scene metadata from _>_ 4,000 hours of synthetic first-order Ambisonics (FOA) mixtures spanning 1—4 sources, background noise, and diverse rooms. Our contributions are threefold: (i) a spatial audio LLM, integrating a spatial encoder, powered by a spatial data and metadata generation pipeline, for comprehensive spatial-scene description; (ii) a permutation-invariant evaluation protocol with 15 metrics that account for multiple sources and environmental attributes; and (iii) extensive experiments demonstrating generalization to real room impulse responses (RIRs), along with careful analyses across SNR, reverberation, source count and other attributes. _Sci-Phi_ advances audio foundation models from isolated object recognition to coherent spatial-scene understanding, with promising results towards real-world generalization. 

### **II. Related Works** 

Research in spatial audio has progressed from sound event detection and localization (SELD) to representation learning and, more recently, to spatial understanding with LLM. Standard SELD systems jointly estimate class labels and locations for multiple sources, with advances in model design [13], [14], training objective [15], and benchmarks [16]. Despite strong ad-hoc performance, they assume a small label set and remain tailored to SELD, limiting openvocabulary and holistic scene understanding. Another line of work explores self-supervised learning (SSL), including contrastive learning [17] and masked reconstruction [18], [19], without the need for labels and therefore learning embeddings that are naturally generalizable to new labels. However, these SSL models were designed and evaluated by task-specific heads for standard SELD and ASR tasks, limiting zero-shot and task co-learning ability. A related direction learns joint spatial audio–text embeddings [20], [21] via CLIP/CLAP-like cross-modal contrastive learning [22], [23]. While useful for captioning and retrieval, these embeddings are generally limited to single sources and are not directly applicable to open-ended text generation, such as multi-source scenery description. 

Audio LLMs, monaural or spatial, define today’s standard of audio foundation models with open-vocabulary user 

**FIGURE 1.** **_Sci-Phi_ architecture, derived from Phi-4-Multimodal (visual components not shown for clarity). Fire and snowflake mark the trainable and frozen components. Light red, blue, and grey colors correspond to spatial, spectral, and textual features, modules, embeddings, and computation flow.** 

queries and responses. They typically pair an audio encoder with a pretrained transformer-decoder language model (i.e., GPT [24]), leveraging strong linguistic priors and a unified next-token objective across tasks such as SELD, ASR, and more general Q&A. Table 1 reviews current spatial audio LLMs to the best of our knowledge; Phi-4-Multimodal [12] is included as a representative monaural audio LLM [7], [8]. Although heterogeneous spatial formats and limited open-sourcing hinder direct apples-to-apples comparison, most existing spatial audio LLMs are restricted to one audio domain (speech or non-speech), provide only partial localization (2-D or without distance), and omit background and room acoustics completely. In contrast, _Sci-Phi_ offers full spatial-scene understanding with a scalable number of directional sources and is the first to demonstrate generalization on real RIRs. 

### **III. Sci-Phi** 

### **_A. Multimodal Features and Architecture_** 

The overall architecture of _Sci-Phi_ is shown in Figure 1. _Sci-Phi_ is a spatial audio LLM with two encoders: a spatial encoder for spatial features and an audio encoder for spectral features. Both features are derived from a first-order Ambisonics (FOA) waveform of four channels ( _W, X, Y, Z_ ), where _W_ is omnidirectional. Concretely, we compute (i) mel spectrograms of all four channels and (ii) intensity vectors (IVs) [25] for ( _X, Y, Z_ ) relative to _W_ . These seven maps (4 mel + 3 IV) are stacked as the spatial features, while the spectral features are the mel spectrogram of the _W_ channel alone, since the monaural audio encoder only accepts singlechannel inputs. 

The spatial encoder borrows the architecture and checkpoint of SELDNet<sup>1</sup> [13], [14]. SELDNet contains 3 convolution layers, 2 gated recurrent units, and 2 self-attention layers. Although the encoder was pretrained on sound event 

> 1Available at https://github.com/partha2409/DCASE2024 <u>seld</u> baseline 

VOLUME , 

2 

detection and localization, both the amount and coverage of training data, with only 13 sound event labels, and mostly only horizontal spatial direction coverage [16], are insufficient to generalize to more complex acoustic scenes (e.g., our test sets). Therefore, we further finetune the spatial encoder together with the LLM by instruction-tuning on a larger and more diverse training set. While separately pretraining the audio encoder on a larger dataset may help generalization even further, we found the joint training of encoder and LLM to perform well. 

We directly use the pretrained monaural audio encoder from Phi-4 Multimodal, which consists of 3 convolution layers and 24 conformer blocks [26]. We freeze the monaural audio encoder to maximally preserve its original audio understanding ability trained on monoaural audio, particularly its state-of-the-art speech recognition capability in 8 languages. Finally, two separate 2-layer linear projectors (trained from scratch for spatial, and frozen for audio encoder) project the spatial and audio encoder outputs to the same dimension (3072) as the text embedding. The spatial, audio, and text embedding are modeled jointly by the LLM Phi-4-Mini [12] (3.8B _small_ LLM). The LLM reads input in the following format: 

"<|user|> **<|spatial|><|audio|>** <|question|><|end|> <|assistant|><|answer|><|end|>" 

where **<|spatial|>** and **<|audio|>** are variablelength placeholders for spatial and audio embeddings, respectively. 

### **_B. Data Generation_** 

Because well-annotated spatial audio corpora large enough to train _Sci-Phi_ are not available publicly, we synthesize firstorder Ambisonics (FOA) training data and paired metadata at scale. Each 10s sample is created by (i) sampling a room with pre-rendered multi-channel room impulse responses (RIRs), (ii) placing 1-–4 directional sound sources distributed in the room and a diffuse background by convolving audio sources with the RIRs, and (iii) randomizing levels, spectral filtering etc. before mixing sources and background. The training set contains 1.6 million 10s mixtures ( _∼_ 4,444h), generated as follows. 

**Rooms and RIRs** . We simulate 10k rooms with the image-source model [27]. Room sizes range from 4×4×3m<sup>3</sup> to 25×25×6m<sup>3</sup> , and the FOA microphone is placed at a random position. For each room we precompute 64 candidate source positions with a roughly spherically uniform direction distribution. We also record room-level attributes such as reverberation time (RT60) and volume. 

**Sound sources** . As diverse sound source corpora we use speech from CommonVoice [28] (8 languages, _∼_ 385h) and general audio from Freesound ( _∼_ 230k files) and the BBC sound-effects collection ( _∼_ 33k files). We clean tags and captions with an LLM to remove recording-condition notes and sound-irrelevant text, and divide them into single-source 

and multi-source/ambient files. We use files described as multi-source/ambient as background noises and convolve them with all 64 RIRs from one room to simulate diffuse sound. 

**Metadata and quantization** . Each mixture is accompanied by human-readable scene metadata. Room fields include RT60 and room volume; background fields include noise type and its loudness; each source has a caption (also transcription for speech), onset/offset times, direction, distance, level (dBA), and C50. To stabilize generation and evaluation, we quantize: (i) 3-D direction (azimuth and elevation) into 26 regions using 45<sup>_◦_</sup> angular bins (e.g., _“upper back-left”, “horizontal front-right”, “above”_ ), (ii) distance to 0.1m, (iii) RT60 and time to 0.1s, (iv) loudness and C50 to 1dB, and (v) room volume to 100m<sup>3</sup> . These choices hit a balance between these physical (and mostly continuous) acoustic attributes and simple and descriptive language targets. The sound levels also allow calculation of SNR. 

**Test sets** . We generate two test sets with 10k clips each (27 h): a held-out **_synthetic-RIR_** test set using 100 unseen rooms and unseen audio sources from SoundBible(.com) and speech from VCTK (English only) [29]; a **_real-RIR_** test set spatializes anechoic sources via real FOA RIRs, and adds real spatial background recordings from 100 real rooms, all from the FOA-MEIR dataset [30]. The FOA-MEIR datasets contains a set of anechoic sound event recordings, and we again use anechoic English speech from VCTK. The FOAMEIR test set is limited in spatial coverage: _no sources outside the horizontal plane_ (above and below _±_ 22 _._ 5<sup>_◦_</sup> ), _no room volume_ information, and _only ambient background noise without specific labels_ . To test these absent conditions, we have to rely on the synthetic test set only. 

### **_C. Training Objective_** 

_Sci-Phi_ is trained to generate a full description of the spatial acoustic scene. We serialize the scene metadata into the <|answer|> string with the template below, starting from the environment to the sources: 

room_volume=<room_volume>; RT60=<rt60>; n_src=<n_src>. noise_label:<noise_type>; noise_loudness=<noise_dB>. Sound label:(time, direction, distance, loudness, C50): <label_1>: (<time_1>, <direction_1>, <distance_1>, <loudness_1>, <C50_1>); <label_2>: (<time_2>, <direction_2>, <distance_2>, <loudness_2>, <C50_2>); ... 

Fields in blue are sample-specific parameters. It is important to note that the **_source enumeration order_** must be fixed a priori for the LLM to learn and will affect its performance (see Table 3). Unless otherwise noted, we order sources by decreasing **_loudness_** . 

The trainable components are the spatial encoder, the spatial projector, and the spatial low-rank adaptation (LoRA) [31] inside the LLM. We keep the existing mono audio LoRA in Phi-4-Multimodal frozen and initialize the 

VOLUME , 

3 

Jiang et al.: 

**TABLE 2. We evaluate multiple metrics on multiple sources in arbitrary orders via either (1) per-metric optimal permutations** _P_ OM **or (2) a single** _P_ OS **that maximizes** _T upleScore_ **(joint** **_What/When/Where_ ) with respect to the target sources. Scene-level metrics (global attribute, no permutation) are in gray.** _P_ OS **scores closely match** _P_ OM **per-metric optima. Note:** _P_ OS **WER may be slightly better than** _P_ OM **because pairs lacking either transcript are skipped.** 

|Protocols<br>RoomVol<br>ErrLog2<br>RT60<br>Err (s)<br>Noise<br>CLAP<br>Co<br>Accura|unt<br>cy (%)<br>Tuple<br>Score|Source<br>CLAP|WER|Direction Accuracy<br>(XYZ_|_ XY _|_ Z, %)|Zone<br>Err (<sup>_◦_</sup>)|Distance<br>ErrRatio|Time<br>IoU|Loudness<br>Err (dB)|C50<br>Err (dB)|
|---|---|---|---|---|---|---|---|---|---|
|**_On synthetic-RIR test set_**||||||||||
|Optimal-Metric _POM_<br>0.590<br>0.092<br>0.662<br>9|1.5<br>0.783|0.694|0.464|85.8 _|_ 92.1 _|_ 94.0|6.1|0.228|0.815|1.011|1.217|
|**Optimal-Source** _POS_<br>0.590<br>0.092<br>0.662<br>9|1.5<br>0.783|0.674|0.449|82.9 _|_ 85.1 _|_ 92.0|8.4|0.258|0.802|1.253|1.348|
|**_On real-RIR test set_**||||||||||
|Optimal-Metric _POM_<br>u.a.<br>0.333<br>u.a.<br>7|5.2<br>0.765|0.712|0.387|79.7 _|_ 90.1 _|_ 88.4|10.4|0.254|0.746|1.642|1.948|
|**Optimal-Source** _POS_<br>u.a.<br>0.333<br>u.a.<br>7|5.2<br>0.765|0.691|0.371|77.4 _|_ 84.4 _|_ 87.5|12.0|0.292|0.737|1.975|2.203|



spatial LoRA with the same configuration, i.e., a rank of 320. We optimize the next-token prediction objective below, with _Q_ denoting the question tokens, _A_ the answer tokens, and _X_ spatial, _X_ spectral the spatial and spectral embeddings: 



_Sci-Phi_ and all baseline models were trained for five epochs with an AdamW optimizer [32], a total batch size of 24, a peak learning rate of 1.0e-4, a linear learning rate warm-up (5% steps) followed by linear decay, on 8 NVIDIA A100 GPUs with bfloat16 precision. 

### **IV. Evaluation Method** 

While many sentence-level NLP metrics calculate a score between the ground-truth and the generated scene description, they miss precision in specific physical or categorical attributes. Therefore, we extract each attribute (e.g., RT60, direction, distance) from the description and calculate ad-hoc metrics on them. The metrics include the cosine similarity of the audio-aware text embedding from **CLAP** [23] for source and noise descriptions, **accuracy** of source counting and direction for which we quantize into 26 XYZ (full sphere), 8 XY (azimuth), and 5 Z (elevation) zones, absolute **error** of direction (with respect to the center of the quantized zone), RT60 ( _s_ ), loudness ( _dB_ ), C50 ( _dB_ ), and transcription (i.e., **WER** ), or the **error ratio** or log2 of it for distance ( _m_ ) and room volume ( _m_<sup>3</sup> ), and finally the intersection-over-union ( **IoU** ) of estimated vs. ground-truth source active intervals. The IoU is defined as 



where _∩_ denotes the intersection given by 



and _t_ on and _t_ off denote the onset and offset time of sound events. A few metrics (room volume, RT60 and background noise type and loudness) are defined and calculated for the entire scene, while others are calculated for each source separately, leading to a critical problem of how to find the 

best matched sources from the generated description to the solution. 

**Permutation-invariant Evaluation.** Language models generate tokens autoregressively in a single output stream. While we train _Sci-Phi_ to enumerate sources by decreasing loudness, different permutations appear still valid to the human perceivers. Therefore, we argue that evaluation should be **_order-invariant_** so that correct answers with mismatched or arbitrary orders are not unfairly penalized. We represent each source with six attributes _(label, time, direction, distance, loudness, C50)_ and parse both the generated description and the reference into lists of tuples, _G_ = [ _g_ 1 _, . . . , gm_ ] and _S_ = [ _s_ 1 _, . . . , sn_ ]. We then seek a permutation matrix _P_ that reorders _G_ (or _S_ ) to calculate Metric( _PG, S_ ) averaged by all sources in the scene. 

We could define the **_optimal-metric permutation_** _POM_ = maxAll _P Metric_ ( _PG, S_ ) that maximizes a single metric, like label or direction. However, the downside is that it ignores cross-attribute association: e.g., if _G_ = [( _dog, left_ ) _,_ ( _cat, right_ )] and _S_ = [( _cat, left_ ) _,_ ( _dog, right_ )], per-metric matching can yield perfect scores for both _label_ and _direction_ despite mismatched association. Instead, we define **_optimal-source permutation_** _POS_ which maximizes a composite _TupleScore_ of multiple attributes and does not advantage any particular metric. 



where the _TupleScore_ is geometric mean of _what_ , _where_ , and _when_ , with each term and the final score normalized to 0 _−_ 1: 

_TupleScore_ ( _g, s_ ) = � _What ·Where ·When_ ( _g, s_ )�1 _/_ 3 (5) with the specific metrics defined as 





where the IoU is given in (2). The _What_ metric uses _WER_ only if a speech source is detected and transcribed 

VOLUME , 

4 



<!-- Start of picture text -->
a“), ; RT60 Error. (s)(s ~ 125Room“ Fy Volume; Error (log2) 0.8oe |* Noise¢ CLAP . 100° Countu Accuracy3 3 $ (%)i if<br>. “ ° g a 4<br>0.0 0.4<br>prii sor?)ELONT(Me:expen’eT) gci-PO pric’ seFT)sebOhi(Mperreni c rY) ” sam pri-a orn)serace™)petxPnr™a sci?i priedCPscPM), pid ©“s(scFO)exon e tttyBehm=<br>Tuple Score 7 a Source CLAP WER Time loU<br>oe i; ; § os t 3 } g 3. $ 0.75 3 0.8 ° ° t<br>ogPriedaeSELONq xPFT)=- sci"—- pried0.4 3 #(SRPT . (SS LON E scr)(MeethFT) Bch Sci-PPO 0.250.50prieda . BenCPM,kiid SC$setecrgcFO)Oye+3: t PO) WSesct+8 i -PM?$6 pried0.4os : FP sch,  pied bs  F sceO eupnettny .ou7 i-Sei-PO  ail ‘<br>so}D irection; ° Accuracy -- XYZA (%)‘ 75{eoDirectioni i Accuracyry 'y - XYi(%)(%@ feDirectioni i  Accuracy - Z to(%) 200: Zong ih<br>*° $ °<br>ji * 4 Foy 5025 . 2 8 2550 ° = ¢ 7 -- . "<br>y vocalizetyizer 4 (st:FT) pri-S1-8 PhPH yzer ) ; R hci PO izer ah<br>\ pried © getONCET) 3 w ror iset SELF  yustseesaeh wygy voealiZe"BOR pried 4 (SCT)SeLONEcc |phiET) wy vores (SCL“we Guerre. Been<br>_ Distance' Error Ratioi . ; Loudness- Error (dB) 2 C50 Error (dB) . | @isrcSynthetiM4 src @1-4 mean<br>; . ° 4 : ; 8 Higher is better ynthetic RIR (solid)<br>“ . : | Real RIR (e<br>= : * . Lower is better<br>oni’— ( 1)geonetttyq 7 — ci?~ ont sor)settace™)awerent’ eeuuphi pre’se6c ¢T)sete rt)phi-aid al+P7 otee src D(14 sre( ona-4 ~<br><!-- End of picture text -->



<!-- Start of picture text -->
$1 Spatial Zone Estimation on Synthetic RIR<br>horizontal front 48:5) 0.04 0.06 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00}<br>horizontal front-left 0.04 BREJo.00 0.01 0.00 0.00 0.00 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.03 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00}<br>horizontal front-right }0.04 0.00 fJo.00 0.03 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00,<br>horizontal left {0.00 0.02 0.00 (9:)} 0.00 0.00 0.04 0.01 0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.01 0.00 0.00 0.00)<br>horizontal right {0.00 0.00 0.03 0.00 [5-78 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.04 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.01 0.00 0.00}<br>horizontal back 40.00 0.00 0.00 0.00 0.00 [15-} 0.03 0.03 0.00 0.00 0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00}<br>horizontal back-left 40.01 0.00 0.00 0.02 0.00 0.05 [9}-1}0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.03 0.00 0.00 0.00)<br>v horizontalupper back-rightfront 40.0040.00 0.00 0.00 0.00 0.000.00 0.00 0.00 0.00 0.05 0.000.00 0.00[Bjo.000.00 9590.04 0.00 0.00 0.00 0.01 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.000.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)0.00}<br>fo} upper front-left {0.00 0.03 0.00 0.01 0.00 0.00 0.00 0.00 0.03 [EF 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)<br>N upper front-right {0.00 0.00 0.09 0.00 0.00 0.00 0.00 0.00 0.04 0.00[iJo.00 0.06 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)<br>oO upper left 70.00 0.00 0.00 0.05 0.00 0.00 0.00 0.00 0.00 0.03 0.00 [8/9 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)<br>a upper right 40.00 0.00 0.00 0.00 0.03 0.00 0.00 0,00 0,00 0,00 0.00 0.00 [MEY 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00<br>Qa upper back 40.00 0.00 0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 [§3:F40.03 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00}<br>wn upper back-left }0.00 0.00 0.00 0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.06 0.00 0.02 [9355}0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)<br>oOi} upper back-right {0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.06 0.00 0.00 0.00 0.00 0.03 0.03 0.00 |¥8-5}0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00}<br>aayis) lower front 40.05 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 [§3-¥§ 0.04 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.02<br>x lower front-left {0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.07 (:790.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00}<br>lower front-right 70.00 0.00 0.13 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.07 0.00 iF 0.00 0.07 0.00 0.00 0.00 0.00 0.00)<br>lower left {0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.05 0.00[E000 0.00 0.02 0.00 0.00 0.00)<br>lower right 70.00 0.00 0.00 0.00 0.09 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.05 0.00 EF o.00 0.00 0.05 0.00 0.00}<br>lower back 40.00 0.00 0.00 0.00 0.00 0.07 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 fo.05 0.07 0.00 0.00<br>lower back-left {0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.07 FF 00 0.00 0.00}<br>lower back-right 70.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.06 0100 FEY 0.00 0.00}<br>above 40.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.02 0.02 0.00 0.04 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 fF 0.00<br>below 40.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.14 0.00 0.00 0.00 0.00 0.00 0.14 0.00 754<br>TeregeSESSLERee eyett TSeeeeeSEETETEETep eeaEESTPSTEREESTeeEETeeeETaeSEREyPSEAHET EYSE BS8<br>2e65BSteRBBsegrseatoxeysastes~€&sstzezeaue2&&seatsSesh55225552252&8ses 2£22e65ssyzageeig252&8582Estimateda58$e5sfeaserSpatial e ervrssereasun8s2$eE ZoneFESg£5s=z 5B66.38ccsRSgzS ¥ws5<br>cR1 Spatial Zone Estimation on Real RIR<br>8 horizontal front {219.01 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)<br>horizontal front-left }0.04o.5 .00 0.05 0.00 0.00 0.01 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.20 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00}<br>© horizontal front-right 70.17 0.00 0.31 0.00 0.04 0.00 0.00 0.00 0.08 0.00 0.34 0.00 0.06 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00)<br>a horizontal left 40.00 0.05 0.00,0.7949 00 0.00 0.05 0.00 0.00 0.00 0.00 0.05 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00}<br>Qa horizontal right 70.00 0.00 0.00 0.04 Ro..00 0.00 0.03 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.05 0.00 0.00 0.00 0.00 0.00)<br>wn horizontal back {0.00 0.00 0.00 0.00 0.00 ERjo.02 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00,<br>wo horizontal back-left 40.00 0.00 0.00 0.08 0.00 0.11 0.00 0.00 0.00 0.00 0.03 0.00 0.02 0.20 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00}<br>Fa]<5 horizontal back-right 40.01SeeegeeeeeeegereeveeigixzeisS252 0.00 0.00 0.00 0.04SP0.07 0.00S550.00 0.00TES0.00 0.00 0.00ESTEE0.00 0.00 0.00 0.00 0.00TET0.00 0.00 0.01EES0.01 0.00 0.38SB 0.00 0.00)8<br>SEEeeET SSE TEER EE SEE EERB EZ SER<br>efeegegelteeseesegeegzseesssgtes<br>SERSSEEE28 eg 2 TAS Ss ee tee FF aS<br>2SsSseeteFfSoc25£§3 £8 £ “fe25658€£ aoa° 8 aoa2 § 86 s2 &6 s2<br><!-- End of picture text -->



<!-- Start of picture text -->
S2 Source Counting on Synthetic RIR R Source Counting on Real RIR<br>3 2.<br>vSa 1 1 0 oO oO gt 0.78 0.19 0.02 0.01<br>Wnfo} wni]fo}<br>‘S24 003 0.01 0 S24 0.05 0.81 0.13 0.01<br>ooore<br>22<br>Ss 3 0.01 0.09 0.89 0.02 Ss 3 0 0.17 0.74 0.09<br>2 2<br>oe=aoe<br>o4 0 0.01 0.18 0.81 0 4 0 0.03 0.28 0.68<br><<<br>a<br>Estimated nNNumber” of Sources+ Estimateda nNNumber™” of Sources+<br><!-- End of picture text -->



<!-- Start of picture text -->
A 0.66 0.66 0.72 Performance0.72074 26.34 vs SNR0.43Level 3.31 2.96<br>0.67 - 0.41 2.72<br>0.57 0.62 0.64 2.90<br>2.33<br>17.80 0.30 2.20<br>14.30<br>low (<10dB)<br>medium (10~20dB)<br>high (=20dB)<br>B Performance vs Reverberation Level<br>0.650.660.66 0.749.7254, 077 q.745 75 15.73 0.51 240 55, 3.23<br>13,343:85 2.12<br>2.30<br>211<br>0.29<br>0.24<br>low (0~0.4s)<br>medium (0.4~0.7s)<br>high (=0.7s)<br>Cc 0.83 Performance0.79 0.74 vs0.78 Same0.79 or11.51Different4336: 0.30Labels0.28of Two1.73 Sources1.74 2.20 2.08<br>0.68<br>Same Labels<br>Different Labels<br>source CLAPPT"rime 100"oUt ie aceXYZ"XYZ" zone EMDrr -+ Err RACtiodaness EWT™ry c50 EIred<br>1.0 1.0 1.0 6 5<br>D 1.0} —e Dir Acc A ©— CLAP Score — Loudness Err<br>|| a DistErrRatio "logo °°) om Time lou 9 5) -m- €50 Err 4<br>el te eZ a ee ee an a) s<br>Sos O68 50.7) a eee 8 9 0? 0.73 FS aS<br>a oe ele gos os $7) | wee le, , | o<br>02 om ory o4 2 ‘<br>° °9 30 60 90 120 150 180° 039 30 60 90 120 150 10> °9 30 60 90 120 150 10<br>Relative Angular Separation of Two Sources (°)<br>1.0 1.0 1.0 6 5<br>E 1.0] —® Dir Acc g| ~e> CLAP Score - ~ Loudness Err<br>" =~ Dist Err Ratio 0.80 °°] a Time lou 895) me C50 Err 4<br>B08So == 06508807 es. a q os073 2,& mig<br>20.6 5 ny © o 3 -— 2<br>Hf |e = | a : gosi os05 3?3*| «—*—*— 8<br>02 2" 04 o4 1 ;<br>2.0 00 02 04 06 08 10 Time00 Intersection03 00 02 over04 Union06 08of Two10 3Sources° 00 02 04 06 08 1.0 °<br>FS10- 12 —®-- DirDist AccErr Ratio 0.80.6 B09© 1.0 —®-= TimeCLAP loUScore = = 10 g— 2.53.0 ‘@—= LoudnessC50 Err Err 3<br>35 * [04 Zo5 508 isri « 0.858 5202.0 23<br>£08) es of) eis) See me |B<br>252. 02 Boel907)gs mf& oseoF 310f8 . ° eo s|ota)<br>as 00 8 g :<br>05 0.4 0.5<br>0.4 -0.2 0A 0<br>) 2 4 >6 0 2 4 >6 oO 2 4 >6<br>Source Duration (s)<br><!-- End of picture text -->

Jiang et al.: 

**TABLE 3. Ablation on the source enumeration order. Results from the synthetic-RIR test set are averaged across 1-4 sources. Only for this ablation, models were trained with 45% (2k hours) data subset. Environmental metrics are colored in gray, for which the source enumeration order has little effect.** 

|Order By<br>RoomVol<br>ErrLog2<br>RT60<br>Err (s)|Noise<br>CLAP|Count<br>Accuracy (%)|Tuple<br>Score|Source<br>CLAP|WER|Direction Accuracy<br>(XYZ_|_ XY _|_ Z, %)|Zone<br>Err (<sup>_◦_</sup>)|Distance<br>ErrRatio|Time<br>IoU|Loudness<br>Err (dB)|C50<br>Err (dB)|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Zone<br>0.639<br>0.104|0.667|**88.1**|0.797|0.665|**0.314**|77.1 _|_ 82.0 _|_ 88.5|11.1|0.279|0.802|1.587|1.606|
|Distance<br>0.632<br>0.104|0.668|87.5|**0.800**|**0.666**|0.323|80.7 _|_ 83.5 _|_ 90.9|9.5|0.284|0.803|1.613|1.621|
|Name<br>0.633<br>0.104|0.666|87.7|0.796|0.658|0.332|80.5 _|_ 83.9 _|_ 90.7|9.5|0.278|0.803|1.665|1.621|
|Onset<br>0.632<br>0.105|0.670|87.7|**0.800**|0.665|0.315|80.3 _|_ **84.0** _|_ 90.6|9.7|0.278|**0.804**|1.624|1.608|
|**Loudness**<br>0.633<br>0.103|0.665|87.6|0.798|0.664|0.329|**80.8** _|_ 83.2 _|_ **91.0**|**9.3**|**0.275**|0.803|**1.579**|**1.579**|



**TABLE 4. Distinct and combined roles of the spatial and spectral features and encoders. Results from the synthetic-RIR test set are averaged across 1–4 sources.** 

|Features|RoomVol|RT60|Noise|Count|Tuple|Source|WER|Direction Accuracy|Zone|Distance|Time|Loudness|C50|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||ErrLog2|Err (s)|CLAP|Accuracy (%)|Score|CLAP||(XYZ_|_ XY _|_ Z, %)|Err (<sup>_◦_</sup>)|ErrRatio|IoU|Err (dB)|Err (dB)|
|Spatial Only|**0.564**|0.097|0.647|87.1|0.699|0.553|1.189|81.7<br>_|_ 84.3<br>_|_ 91.0|9.5|0.269|0.760|1.626|1.507|
|Spectral Only|0.868|0.654|0.631|25.0|0.562|0.285|0.464|9.1 _|_ 15.8 _|_ 46.5|75.5|0.417|0.589|7.893|6.460|
|Spatial+Spectral|0.590|**0.092**|**0.662**|**91.5**|**0.783**|**0.674**|**0.449**|**82.9** _|_ **85.1** _|_ **92.0**|**8.4**|**0.258**|**0.802**|**1.253**|**1.348**|





<!-- Start of picture text -->
Count Accuracy<br>100.0<br>98.4 95.8<br>99.8 95.1 87.4<br>99.7 96.4 88.8 80.9<br>99.8 96.9 91.4 83.0 87.8<br><!-- End of picture text -->



<!-- Start of picture text -->
Source CLAP<br>0.710<br>0.723 0.663<br>0.730 0.676 0.653<br>0.733 0.677 0.655 0.633<br>0.733 0.684 0.656 0.636 0.621<br><!-- End of picture text -->



<!-- Start of picture text -->
Direction Accuracy - XYZ<br>1 84.1<br>2 86.2 82.3<br>3 86.1 83.3 80.0<br>4 87.9 84.6 81.1 78.1<br>5 89.1 85.7 82.6 79.7 77.8<br>Max # Sources Trained on<br><!-- End of picture text -->



<!-- Start of picture text -->
Distance Error Ratio<br>0.303<br>0.271 0.261<br>0.272 0.247 0.268<br>0.257 0.239 0.258 0.279<br>0.252 0.229 0.263 0.278 0.261<br><!-- End of picture text -->



<!-- Start of picture text -->
Time IoU<br>0.843<br>0.853 0.820<br>0.852 0.822 0.795<br>0.844 0.812 0.790 0.762<br>0.838 0.812 0.783 0.757 0.755<br><!-- End of picture text -->

**FIGURE 6.** **_Sci-Phi_ is scalable to the number of sources: still performs strongly up to 5 sources, and preserves performance on fewer sources.** 

taxonomies. The current framework also assumes stationary sources; generating per-timestep trajectories with an LLM would be computationally very expensive. We leave these directions to future work. 

### **REFERENCES** 

- [1] Jens Blauert, _Spatial Hearing: The Psychophysics of Human Sound Localization_ , The MIT Press, 10 1996. 

- [2] Andrew J. Kolarik, Brian C. J. Moore, Pavel Zahorik, Silvia Cirstea, and Shahina Pardhan, “Auditory distance perception in humans: a review of cues, development, neuronal bases, and effects of sensory loss,” _Attention, Perception & Psychophysics_ , vol. 78, pp. 373 – 395, 2015. 

- [3] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , vol. 28, pp. 2880–2894, 2020. 

- [4] Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen, “Joint measurement of localization and detection of sound events,” in _2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)_ , 2019, pp. 333–337. 

- [5] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large- 

scale weak supervision,” in _International conference on machine learning_ . PMLR, 2023, pp. 28492–28518. 

- [6] Tara N. Sainath, Ron J. Weiss, Kevin W. Wilson, Bo Li, Arun Narayanan, Ehsan Variani, Michiel Bacchiani, Izhak Shafran, Andrew Senior, Kean Chin, Ananya Misra, and Chanwoo Kim, “Multichannel signal processing with deep neural networks for automatic speech recognition,” _IEEE/ACM Trans. on Audio, Speech, and Language Processing_ , vol. 25, no. 5, pp. 965–979, 2017. 

- [7] Yuan Gong, Hongyin Luo, Alexander H Liu, Leonid Karlinsky, and James Glass, “Listen, think, and understand,” in _Intl. Conf. on Learning Representations_ , 2024. 

- [8] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou, “Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models,” _arXiv preprint arXiv:2311.07919_ , 2023. 

- [9] Zhisheng Zheng, Puyuan Peng, Ziyang Ma, Xie Chen, Eunsol Choi, and David Harwath, “Bat: Learning to reason about spatial sounds with large language models,” in _International Conference on Machine Learning_ , 2024, pp. 61454–61469. 

- [10] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Jun Zhang, Lu Lu, Zejun Ma, Yuxuan Wang, et al., “Can large language models understand spatial audio?,” in _Proc. Interspeech 2024_ , 2024, pp. 4149–4153. 

- [11] Ayushi Mishra, Yang Bai, Priyadarshan Narayanasamy, Nakul Garg, and Nirupam Roy, “Sing: Spatial context in large language model for next-gen wearables,” in _Forty-second International Conference on Machine Learning_ . 

VOLUME , 

8 

- [12] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al., “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixtureof-loras,” _arXiv preprint arXiv:2503.01743_ , 2025. 

- [13] Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen, “Sound event localization and detection of overlapping sources using convolutional recurrent neural networks,” _IEEE Journal of Selected Topics in Signal Processing_ , vol. 13, no. 1, pp. 34–48, 2018. 

- [14] Sooyoung Park, Youngho Jeong, and Taejin Lee, “Self-attention mechanism for sound event localization and detection,” in _DCASE2021 Challenge — Techn. Reports_ , 2021, pp. 1–4, Task 3: Sound Event Localization and Detection with Directional Interference. 

- [15] Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Naoya Takahashi, Emiru Tsunoo, and Yuki Mitsufuji, “Multi-accdoa: Localizing and detecting overlapping sounds from the same class with auxiliary duplicating permutation invariant training,” in _International Conference on Acoustics, Speech and Signal processing (ICASSP)_ . IEEE, 2022, pp. 316–320. 

Tyers, and Gregor Weber, “Common voice: A massively-multilingual speech corpus,” in _Proceedings of the Twelfth Language Resources and Evaluation Conference_ , 2020, pp. 4218–4222. 

   - [29] Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald, “CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019. 

   - [30] Masahiro Yasuda, Yasunori Ohishi, and Shoichiro Saito, “Echoaware adaptation of sound event localization and detection in unknown environments,” in _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2022, pp. 226–230. 

   - [31] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al., “Lora: Low-rank adaptation of large language models.,” _ICLR_ , vol. 1, no. 2, pp. 3, 2022. 

   - [32] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay regularization,” in _Intl. Conf. on Learning Representations_ . 

   - [33] John S Bradley, Hiroshi Sato, and Michel Picard, “On the importance of early reflections for speech in rooms,” _The Journal of the Acoust. Soc. of America_ , vol. 113, no. 6, pp. 3233–3244, 2003. 

- [16] Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel A Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, et al., “Starss23: An audio-visual dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events,” _Adv. in neural information proc. systems_ , vol. 36, pp. 72931–72957, 2023. 

- [17] Xilin Jiang, Cong Han, Yinghao Aaron Li, and Nima Mesgarani, “Exploring self-supervised contrastive learning of spatial sound event representation,” in _ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2024, pp. 1281–1285. 

- [18] Antoni Dimitriadis, Siqi Pan, Vidhyasaharan Sethu, and Beena Ahmed, “Spatial hubert: Self-supervised spatial speech representation learning for a single talker from multi-channel audio,” _arXiv preprint arXiv:2310.10922_ , 2023. 

- [19] Goksenin Yuksel, Marcel van Gerven, and Kiki van der Heijden, “General-purpose audio representation learning for real-world sound scenes,” _arXiv preprint arXiv:2506.00934_ , 2025. 

- [20] Bhavika Devnani, Skyler Seto, Zakaria Aldeneh, Alessandro Toso, Elena Menyaylenko, Barry-John Theobald, Jonathan Sheaffer, and Miguel Sarabia, “Learning spatially-aware language and audio embeddings,” _Adv. in Neural Information Proc. Systems_ , vol. 37, pp. 33505–33537, 2024. 

- [21] Jinbo Hu, Yin Cao, Ming Wu, Feiran Yang, and Jun Yang, “SALM: Spatial audio language model with structured embeddings for understanding and editing,” 2025. 

- [22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in _Intl. Conf. on Machine Learning_ , 2021, pp. 8748–8763. 

- [23] Benjamin Elizalde, Soham Deshmukh, and Huaming Wang, “Natural language supervision for general-purpose audio representations,” in _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2024, pp. 336–340. 

- [24] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al., “Language models are few-shot learners,” _Advances in neural information processing systems_ , vol. 33, pp. 1877–1901, 2020. 

- [25] Masahiro Yasuda, Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, and Keisuke Imoto, “Sound event localization based on sound intensity vector refined by dnn-based denoising and source separation,” in _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2020, pp. 651–655. 

- [26] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, et al., “Conformer: Convolution-augmented transformer for speech recognition,” in _Proc. Interspeech 2020_ , 2020, pp. 5036–5040. 

- [27] Jont B Allen and David A Berkley, “Image method for efficiently simulating small-room acoustics,” _The Journal of the Acoustical Society of America_ , vol. 65, no. 4, pp. 943–950, 1979. 

- [28] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis 

VOLUME , 

9 


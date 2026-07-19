IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

63 

# SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing 

Ziyang Ma , Guanrou Yang, Wenxi Chen, Zhifu Gao, Yexing Du , Xiquan Li, Zhisheng Zheng, Haina Zhu , Jianheng Zhuo, Zheshu Song, Ruiyang Xu, Tiranrui Wang , Yifan Yang , Yanqiao Zhu , Zhikang Niu _, Student Member, IEEE_ , Liumeng Xue _, Member, IEEE_ , Yinghao Ma, Ruibin Yuan, Shiliang Zhang , Kai Yu _, Fellow, IEEE_ , Eng Siong Chng _, Senior Member, IEEE_ , and Xie Chen _, Senior Member, IEEE_ 

**_Abstract_ —The recent surge in open-source Multimodal Large Language Models (MLLM) frameworks, such as LLaVA, provides a convenient kickoff for artificial intelligence developers and researchers. However, most of the MLLM frameworks take vision as the main input modality, and provide limited in-depth support for the modality of speech, audio, and music. This situation hinders the development of audio-language models, and forces researchers to spend a lot of effort on code writing and hyperparameter tuning. We present SLAM-LLM, an open-source deep learning framework designed to train customized MLLMs, focused on speech, language, audio, and music processing. SLAM-LLM provides a modular configuration of different encoders, projectors, LLMs, and parameter-efficient fine-tuning plugins. SLAM-LLM also includes** 

Received 1 July 2025; revised 19 December 2025; accepted 8 January 2026. Date of publication 12 January 2026; date of current version 18 February 2026. This work was supported in part by the National Natural Science Foundation of China under Grant U23B2018, in part by Shanghai Municipal Science and Technology Major Project under Grant 2021SHZDZX0102, and in part by Yangtze River Delta Science and Technology Innovation Community Joint Research Project under Grant 2024CSJGG01100. _(Corresponding author: Xie Chen.)_ 

Ziyang Ma is with the X-LANCE Lab, School of Computer Science, MoE Key Lab of Artificial Intelligence Shanghai Jiao Tong University, Shanghai 200240, China, and also with Nanyang Technological University, Singapore 639798. 

Guanrou Yang, Wenxi Chen, Xiquan Li, Haina Zhu, Jianheng Zhuo, Zheshu Song, Ruiyang Xu, Yifan Yang, Yanqiao Zhu, Zhikang Niu, and Kai Yu are with the X-LANCE Lab, School of Computer Science, MoE Key Lab of Artificial Intelligence Shanghai Jiao Tong University, Shanghai 200240, China. 

ZhifuGaoandShiliangZhangarewithTongyiLab,AlibabaGroup,Hangzhou 310052, China. 

Yexing Du is with Peng Cheng Laboratory, Guangdong 518055, China. 

Zhisheng Zheng is with the The University of Texas at Austin, Austin, TX 78712 USA. 

Tiranrui Wang is with Tianjin University, Tianjin 300072, China. 

Liumeng Xue and Ruibin Yuan are with the The Hong Kong University of Science and Technology, Hong Kong, SAR, China. 

Yinghao Ma is with the Queen Mary University of London, E1 4NS London, U.K. 

Eng Siong Chng is with Nanyang Technological University, Singapore 639798. 

Xie Chen is with X-LANCE Lab, School of Computer Science, MoE Key Lab of Artificial Intelligence Shanghai Jiao Tong University, Shanghai 200240, China, and also with Shanghai Innovation Institute, Shanghai 200433, China (e-mail: chenxie95@sjtu.edu.cn). 

Open source at https://github.com/X-LANCE/SLAM-LLM. Digital Object Identifier 10.1109/JSTSP.2026.3653157 

**detailed training and inference recipes for mainstream tasks, along with high-performance checkpoints like LLM-based Automatic Speech Recognition (ASR), Automated Audio Captioning (AAC), and Music Captioning (MC). Some of these recipes have already reached or are nearing state-of-the-art performance, and some relevant techniques have also been accepted by academic papers. We hope SLAM-LLM will accelerate iteration, development, data engineering, and model training for researchers. We are committed to continually pushing forward audio-based MLLMs through this open-source framework, and call on the community to contribute to the LLM-based speech, audio and music processing.** 

**_Index Terms_ —Multimodal large language model (MLLM), large language model, framework, toolkit, speech processing, language processing, audio processing, music processing.** 

## I. INTRODUCTION 

HE rapid advancement of Large Language Models (LLMs) **T** has catalyzed the development of multimodal learning systems that integrate various forms of input such as text, vision, as well as audio. While recent open-source frameworks, such as LLaVA [1] and OpenFlamingo [2] have demonstrated remarkable capabilities in vision-language modeling, they remain largely vision-centric. These frameworks offer limited support for non-visual modalities, particularly in the areas of speech, audio, and music processing. This lack of native support presents significant challenges for researchers working on auditory tasks, often forcing them to retrofit vision-based systems through cumbersome adaptations, resulting in inefficiencies and fragmented development workflows. 

To address these limitations, we introduce **SLAM-LLM: a modular, open-source framework for Multimodal Large Language Models (MLLMs), with a specific focus on speech, audio, and music.** SLAM-LLM is designed to lower the barrier for constructing, training, and deploying LLM-based systems that process audio-related modalities. Its core architecture follows a clean encoder–projector–LLM modular design, enabling seamless customization of model components through YAML configuration files. The framework supports a wide range of 

© 2026 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/ 

64 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

pretrained encoders (e.g., Whisper [3], HuBERT [4], BEATs [5], MERT [6]), projection modules (MLP layers, CNN layers, Q-Former [7]), and LLM backbones (e.g., LLaMA [8], [9], Vicuna [10], Qwen [11]). It also incorporates parameter-efficient fine-tuning (PEFT) strategies such as Low-Rank Adaptation (LoRA) and prefix-tuning, making it suitable for low-resource and domain-specific adaptation. SLAM-LLM’s design also addresses common engineering challenges in multimodal modeling, such as handling variable-length inputs and efficient finetuning, while supporting rapid prototyping and reproducibility. 

Beyondarchitecturalflexibility,SLAM-LLMprovidesacomprehensive suite of training recipes and pretrained checkpoints for a variety of key tasks: Automatic Speech Recognition (ASR), Speech-to-Text Translation (S2TT), Speech Emotion Captioning (SEC), Automated Audio Captioning (AAC), and Music Captioning (MC). Extensive experiments across these tasks reveal a series of key insights. 

In brief, SLAM-LLM makes several significant contributions: 

- 1) A unified, flexible, and extensible framework for developing LLM-based models for audio-related modalities; 

- 2) Extensive modular support for diverse encoders, projectors, and large language models; 

- 3) A library of curated training and inference recipes across speech, audio, and music tasks; 

- 4) State-of-the-art results across benchmarks, particularly in speech recognition and audio captioning; 

- 5) Open-source implementation encouraging community collaboration and rapid innovation. 

SLAM-LLM bridges a major gap in the current MLLM ecosystem and promotes the development for future progress in general-purpose audio-language models (ALMs). By opensourcing the framework and accompanying recipes, SLAMLLM invites the broader community to contribute to and accelerate innovation in this important yet limitedly developed area of multimodal AI. 

## II. MULTIMODAL LARGE LANGUAGE MODEL FRAMEWORK 

The development of LLMs has catalyzed a rapidly expanding ecosystem of open-source toolkits designed to support the training, fine-tuning, and deployment of LLMs across a variety of modalities. In particular, the recent surge of interest in multimodal learning has motivated the emergence of several frameworks that extend LLMs beyond pure text understanding. 

Among these toolkits, a number of general-purpose LLM infrastructure frameworks have been proposed to facilitate model development and customization. Notably, LLaMA-recipes [12], an official collection of implementations and usage patterns for the LLaMA model family released by Meta, provides end-toend support for both text-based and visual-language models. LLaMA-Adapter [13] introduces a parameter-efficient tuning strategy for adapting frozen LLaMA models to instructionfollowing and, in its extended version, visual modalities, by introducing a vision-language alignment mechanism. LLaMA2Accessory [14] further generalizes this line of work, offering a comprehensive toolbox for pretraining, fine-tuning, and inference across both unimodal and multimodal settings, with native support for integrating visual encoders such as CLIP [15] 

and DINOv2 [16]. In parallel, LLaMA-Factory [17] provides a unified and extensible training interface for over 100 foundation models, including multimodal branches that support image, audio, and video tasks via task-specific adapters and fine-tuning templates. Additionally, LMFlow [18] and LitGPT offer efficient solutions for finetuning and inference, focusing on memory optimization, scalability, and support for multimodal inputs. 

In contrast to these infrastructure-focused toolkits, another line of work has produced end-to-end multimodal LLMs designed specifically for vision-language models (VLMs). MiniGPT-4 [19] demonstrates that aligning a pretrained visual encoder with a frozen language model using a lightweight projection layer enables powerful vision-language dialogue capabilities. LLaVA [1] adopts a two-stage training strategy that combines visual feature alignment with instruction tuning, resulting in a highly capable open-source visual dialogue model. TinyLLaVA [20] builds on this approach with a focus on model efficiency, achieving competitive multimodal performance with significantly reduced parameter counts. Meanwhile, OpenFlamingo [2] reimplements DeepMind’s Flamingo [21] architectureforopenaccess,supportingfew-shotlearningacross sequences of interleaved image-text pairs and offering pretrained models at various scales. 

However, most existing toolkits primarily target text or vision tasks, offering limited or no support for non-visual modalities such as speech, audio, and music. As a result, researchers in auditory domains—e.g., speech recognition, audio captioning, or music analysis—must often repurpose tools whose architectures and pipelines are tightly coupled to vision or text, requiring substantial adaptation effort. This lack of native support hinders progress in building efficient audio-language models. While toolkits like ESPnet<sup>1</sup> focus on end-to-end speech systems and Fairseq<sup>2</sup> focus on speech sequence modeling, they do not adopt an LLM-centric design, which is a paradigm shift in SLAMLLM. The framework addresses this gap with a modular design that seamlessly integrates speech, audio, and music encoders with LLMs. In SLAM-LLM, all auditory tasks are unified into an auto-regressive generation process, allowing researchers to leverage the strong text power of LLMs while modularly incorporating proven speech processing best practices to guide the generative performance. Its flexible architecture supports easy customization of encoders, projectors, and LLMs, and is compatible with PEFT [22], FSDP [23], and DeepSpeed [24]. With comprehensive training and inference recipes, SLAMLLM accelerates research and promotes community-driven development via its open-source release. 

## III. DESIGN OF SLAM-LLM 

## _A. Overview of SLAM-LLM_ 

SLAM-LLM allows users to configure customized MLLMs through a YAML file, significantly reducing the burden on model construction. Users specify the necessary model components, training strategies, and data formats in the YAML file, enabling 

1[Online]. Available: https://github.com/espnet/espnet 

2[Online]. Available: https://github.com/facebookresearch/fairseq 



<!-- Start of picture text -->
glIIIIIIIIIIII22III4 gTIDIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIILILII2II<br>ornare sneer ese a<br>! YAML File rw ' Customized Recipe mw<br>:!  ModelConfig. ut=! '' pAcPAacgaPGA GN wes taae<br>' sailor-7B a Model Zoo ' : \ 7 \ 4 " uy<br>: Linear Projector i * Qwen H 1 i ‘ ha<br>‘ ia T tinear Projector i VA TABLA te<br>' Whisper Large-v3 re + QFormer ' iw a iw ree<br>''' - iaut ++ BEATSHuBeRT —1——' > Projector ry tate<br>: Training Config tiei SS+ Mug '' LLM nitta<br>‘ LoRA ut ~ ' oka ua<br>: srt i ———+ Encoder te<br>: Me Datasets ' te<br>‘: Data Config ti a py ceatcpeedy ' ia<br>! Eneacn Dataset mo Sees+ Gigaspeech H1 | USER: teflon Transcribe speechto text. ASSISTANT: {T}. tame<br>: <1 *clothe —_ ue<br>' Gan ue Sema | Speech/Audio/Music Prompt T: Transcript ua<br>H i + Lp-Musiccaps H me<br><!-- End of picture text -->



<!-- Start of picture text -->
Encoder um<br>(0 PEFT Pye (PERT Pye<br>/I|LORA,Speech Adapter, Encoder Prefix-Tuning .. \1H1 LORA,1!  Adapter, Prefix-Tuning ... ‘iI!<br>!iI| LanguageHSIAO Encoder rNiyi44 i'raVariable Length Projection eN7|H fotftwot_-t LOMA, Vicuna, Qwen,ChatGLM |1i!<br>|i G2P, GPT-2, TinyLlama, Llama 4.4,io 1 | Linear, CNN 'i ofi |1 Codec Language Model |'<br>! “S30 peetianoaon te VALL-E, VALLE-X. |<br>| Audio Encoder 1| 4ee| ( Fixed Length Projection ron1 yt ij<br>| BEATS,EAT, Spatiol-AST, CLAP -1-""" | | @-Former H At '<br>II| Music Encoder 11iif of i‘Ne oo------ +--+ += eee? Cl} ‘|JeH! SumbolicMuPT Music LanguageModel |!H!<br>|t%,‘. MERT, MusicM,MuQ os+/Hi '‘‘‘. a/1!<br><!-- End of picture text -->

66 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

TABLE I 

DIFFERENT COMBINATIONS OF SUPERVISED SPEECH ENCODERS AND LLMS TO CONDUCT LLM-BASED ASR. WE SHOW RESULTS OF WHISPER MODELS WITH DIFFERENT SIZES ON LLMS OF DIFFERENT SCALES 



feeding into the LLM. The projector uses a hidden size of 2048; encoder output and LLM input dimensions vary by model. The training recipes and model checkpoints can be found here<sup>3</sup> . To address the practical requirements for real-world deployment, we report the compute resources for a representative task using a 7B-scale LLM backbone: while training configurations may vary slightly between experiments to ensure peak performance, the underlying resource requirements remain similar. Training an LLM-based model on 1,000 hours of speech data for 1 epoch, while keeping the LLM and encoder frozen and updating only the projector, requires approximately 4 hours using 4 _×_ NVIDIA A100 (80 GB) GPUs. 

_b) Datasets:_ We use the LibriSpeech [30] benchmark with 960 hours of training data, without augmentation or splicing. Validation and testing are conducted on the 10-hour dev-other, test-clean, and test-other subsets. 

_c) Experiments on different supervised speech encoders:_ Table I presents ASR results across various combinations of supervised speech encoders and LLMs, highlighting several key findings. First, ASR performance improves with larger Whisper encoders. For example, using TinyLLaMA-1.1B, the test-clean WER drops from 12.72 (Whisper-tiny) to 4.39 (Whisper-large), showing that larger encoders better capture speech features. However, returns diminish with size: the WER improvement from Whisper-medium (4.65) to Whisper-large (4.39) is marginal. Second, chat models consistently outperform pre-trained LLMs across all encoder sizes. For instance, with Whisper-large, LLaMA-2-7B yields a test-clean WER of 3.01, while LLaMA-2-Chat-7B improves to 2.72. This suggests SFT enhances the model’s ability to treat speech embeddings as a form of “language”, benefiting from translation-like learning during fine-tuning. Finally, with Whisper-large fixed, Vicuna-7B achieves the best results, reaching a 2.58 WER on test-clean— surpassing both LLaMA-2-7B (3.01) and LLaMA-2-Chat-7B (2.72). This indicates that Vicuna, fine-tuned on user-shared conversational data, generalizes more effectively. 

_d) Experiments on different self-supervised speech encoders:_ Table II analyzes self-supervised speech encoders for LLMbased ASR, using Vicuna-7B-v1.5 as the fixed LLM. WER generally improves with larger encoder sizes, mirroring trends observed with supervised models. At the base scale ( 95 M parameters, 768 hidden size), HuBERT and WavLM perform on par with Whisper-small, showing no clear advantage for selfsupervised learning. However, at 300 M parameters, WavLM 

> 3[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/asr_librispeech 

TABLE II 

EXPLORE THE PERFORMANCE WITH DIFFERENT SSL SPEECH ENCODERS FOR LLM-BASED ASR. LS-960 MEANS THE LIBRISPEECH 960 HOURS DATASET. FT DONATES FINE-TUNING 



Large surpasses all supervised encoders in TableI, including Whisper-medium (300 M) and Whisper-large (600 M), while HuBERT’s gains from Base to Large are more modest. Finetuning HuBERT Large on LibriSpeech yields substantial improvements, achieving 2.10% WER on test-clean and 4.26% on test-other—outperforming WavLM Large. Scaling further to HuBERT X-Large ( 1B parameters, fine-tuned) delivers the best results: 1.84% WER on test-clean and 3.39% on testother, representing relative reductions of 28.7% and 47.4% over Whisper-large. These findings underscore the strong benefits of scaling and fine-tuning SSL encoders for LLM-based ASR. 

_e) Comparison with Non-LLM-based ASR models:_ Table III compares our LLM-based ASR model (SLAM-LLM) with state-of-the-art NN-based systems, including specialist models trained on LibriSpeech-960, self-supervised models pre-trained on large-scale unlabeled speech, and universal models trained on multilingual labeled datasets. For LibriSpeech-960 specialist models, we compare against ContextNet [35], Conformer [36], Branchformer [37] (ESPnet), and Zipformer [38] (K2). These models use extensive system-level techniques—SpecAugment, speed perturbation, EMA averaging—and often fuse in-domain LMs trained on LibriSpeech text. Despite forgoing such engineering, our LLM-based model outperforms the best of these specialist systems. However, SSL models like HuBERT-large/xlarge implemented with Fairseq and WavLM-large implemented with UniSpeech, pre-trained on larger unlabeled datasets and paired with in-domain LMs, outperform our model on the noisy test-other subset. These gains are largely attributable to LM integration, suggesting that domain-specific LLM fine-tuning may offer similar improvements, albeit at the cost of generalization. Compared to general-purpose models, our system surpasses Whisper-large-v2, the stronger Whisper-large-v3, and OWSM-v3.1, a top academic baseline. These results highlight 

67 

MA et al.: SLAM-LLM: A MODULAR, OPEN-SOURCE MULTIMODAL LARGE LANGUAGE MODEL FRAMEWORK 

### TABLE III 

COMPARED TO PRIOR NN-BASED MODELS ON LIBRISPEECH, _SPECIALIST MODELS_ ARE TRAINED SOLELY ON THE 960-HOUR LIBRISPEECH DATASET, OFTEN USING _IN-DOMAIN LMS_ BUILT FROM LIBRISPEECH TEXT AND TRANSCRIPTS. _SSL-BASED MODELS_ ARE SELF-SUPERVISED MODELS PRE-TRAINED ON LARGE-SCALE UNLABELED DATA, THEN FINE-TUNED WITH SUPERVISION. _UNIVERSAL MODELS_ REFER TO GENERAL-PURPOSE SYSTEMS TRAINED ON MASSIVE LABELED SPEECH-TEXT PAIRS. REPOSITORY LINKS ARE PROVIDED FOR REPRODUCIBILITY. “OURS” DENOTES RESULTS FROM OUR BEST SYSTEM IN TABLE II 



TABLE IV 

COMPARISON OF WHISPER AND LLM-BASED ASR IN WER/CER FOR LOW RESOURCE LANGUAGES AMONG THAI(TH), VIETNAMESE(VI), AND INDONESIAN(ID). “+ FINE-TUNING” REFERS TO THE MODEL THAT IS FINE-TUNED USING THE SAME AMOUNT LOW-RESOURCE LANGUAGE TRAINING DATA COMPARED TO THE LLM-BASED ASR MODELS 



the effectiveness of our LLM-based approach, achieving competitiveorsuperiorASRperformancewithoutheavyengineering or massive pre-training. 

_f) Experiments on minority language datasets:_ Beyond highresource languages like English and Chinese, low-resource speech recognition remains a major challenge. This experiment targets minority Southeast Asian languages: Thai (th), Vietnamese (vi), and Indonesian (id), using GigaSpeech 2 [26] for both training and evaluation. To mimic real-world lowresource conditions, training data is limited to 200 hours per language. Our model uses the Whisper large-v3 encoder [3], ReLU-activated MLP layers as the projector, and Sailor-7B [25] as the LLM. To improve performance, we partially fine-tune the encoder by freezing its first 30 layers and updating only the final two. For the LLM, we apply LoRA adapters to the query and value projections in each self-attention layer. Table IV reports 

TABLE V 

COMPARISON OF WHISPER LARGE-V3 AND LLM-BASED ASR IN MANDARIN-ENGLISH CODE SWITCHING ASR. MER DENOTES MIXED ERROR RATE FOR BOTH CHINESE CHARACTER AND ENGLISH WORDS 

results on low-resource ASR tasks. For Thai and Vietnamese, the LLM-based model (Whisper encoder + Sailor-7B) outperforms both Whisper and Whisper-finetune. In Indonesian, while it falls short of Whisper-finetune, it still surpasses the base Whisper model. These results suggest that incorporating LLMs effectively leverages rich textual knowledge to improve ASR under limited-resource settings. Sailor-7B, fine-tuned for Southeast Asian languages, further enhances performance. Compared to Vicuna-based models, the Sailor-based system achieves better results across all three languages, underscoring the value of language-specific LLM selection in low-resource scenarios. 

_g) Experiments on the code switching dataset:_ We also conducted experiments on the ASRU 2019 Mandarin-English CodeSwitching Challenge dataset [41], which includes 200 hours of code-switching speech and 500 hours of Mandarin-only speech. In this study, we use only the 200-hour code-switching subset for training and a 20-hour test set for evaluation. Our system uses Whisper large-v3 as the speech encoder, Qwen2-7B as the LLM decoder, and a lightweight linear projector to align speech and text modalities. To enhance performance, LoRA adapters are applied to the query and value matrices in each self-attention layer of the LLM. Table V shows that the LLM-based model outperforms Whisper large-v3 across all metrics on the codeswitching test set, achieving a 20.2% relative reduction in MER, highlighting the effectiveness of the LLM-based architecture. 

## _B. Contextual Automatic Speech Recognition (CASR)_ 

_1) Task Setup:_ We investigate two types of Contextual ASR tasks: **Visual Contextual Speech Recognition** and **Contextual Biasing Speech Recognition** , as shown in Fig. 3. In Visual Contextual ASR, textual keywords extracted from presentation slides are used to improve transcription accuracy for conference content. Each speech segment is paired with pre-processed OCR results and slide-specific keywords. In Contextual Biasing ASR, a predefined list of hundreds to thousands of biasing terms—such as named entities, contact names, or song titles—is provided to help recognize rare or domain-specific vocabulary. 

## _2) Visual Contextual Speech Recognition:_ 

_a) Datasets:_ We use the SlideSpeech [42] dataset for training and evaluation. This large-scale audio-visual corpus is constructedfromYouTubeconferencevideosandprovideshighquality transcribed speech aligned with synchronized slides. It includes 720p videos, 16 kHz audio, pre-processed OCR results, and extracted keywords per segment, supporting multimodal ASR tasks. Two training sets are available: a large set (L95) with 473 hours and a smaller subset (S95) with 161 hours. The development and test sets contain 5.07 and 8.75 hours, respectively. The dataset’s alignment between speech and slides 

68 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

|(a) Visual ContextualASR<br>nf<br>LLM<br>H<br>‘|fy EOS><br>{|(b) Contex<br>I|tual BiasingA|SR|LLM|NA<br>Ay EOs><br>\<br>i<br>\|
|---|---|---|---|---|---|---|
|Projector &<br>LLMue<br>“<br><br>|“<br>-|Proje<br>|ctor<br>6||LLMToke|nizer<br>.<br>:<br>.<br><br>|
|<br><br>|||||||
|<br>ofteftncngie<br>cone<br>=<br>strata €<br>Slide<br>Fig. 3.<br>(a) LLM-based Visual Contextual ASR inSLAM-<br>ASR RESULTS OFWITH OR WITHOUT CONTEXTUAL KEYWORDS<br>PR<br><br><br>|LLM framew<br>TAB<br> EVALUATED<br>EVIOUS NN-<br>|\<br>efefti<br>'<br>ork. (b) LLM<br>LE VI<br> ON DEV/TE<br>BASED MOD<br>|erifie<br>-based Con<br>ST DATASETS<br>ELS|textual B<br>, TRAINE|eecourr <br>iasingASR <br>D ON S95/L9<br>|2<br>HotwordsList<br>in SLAM-LLM framework.<br>5 DATASETS, COMPARED WITH<br>|
|Train<br>Model<br>Contextual<br><br>||ev<br>||||est<br><br>|
|y<br>WER|B-WER-|U-WER_|Recall<br>ft|WER|B-WER-|U-WER<br>Recall ft|
|SlidesSpeech [42]<br>x<br>21.05|31.27|20.29|68.76|21.22|26.60|20.83<br>73.51|
|505<br>CPP[42]<br>v<br>20.80<br><br><br><br>|28.61<br>|20.22<br>|71.48<br>|20.95<br>|24.05<br>|20.73<br>76.10<br><br>|
|<br><br><br>(61h)<br>LCB-net[43]<br>v<br>18.80|27.90|18.11|72.09|19.21|23.70|18.89<br>76.48|
|<br><br><br>Ours<br>x<br>11.57|16.23|11.28|83.83|11.80|13.52|11.71<br>86.71|
|Ours<br>v<br>11.14|8.92|11.36|91.44|11.26|7.67|11.52<br>92.50|
|SlidesSpeech [42]<br>x<br>13.09|16.13|12.87|83.90|12.89|12.70|12.90<br>87.43|
|CPP [42]<br>v<br>12.64|12.39|12.66|87.64|12.38|9.32|12.60<br>90.86|
|L95<br>LCB-net[43]<br>v<br>12.21|12.12|12.21|87.98|12.02|9.03|12.24<br>91.12|
|<br><br><br>(473h)<br>Ours<br>x<br>9.38|11.98|9.19|88.08|9.34|9.52|9.33<br>90.64|
|+ LoRA<br>x<br>8.82|9.62|8.77|90.38|8.61|7.34|8.72<br>92.84|
|Ours<br>v<br>8.91|6.07|9.13|94.02|9.14|5.47|9.42<br>94.87|
|+LoRA<br>v<br>8.30|5.22|8.53|94.87|8.46|4.89|8.73<br>95.31|



makes it well-suited for text-enhanced multimodal ASR, partic- _ transcription. Fine-tuning the LLM with LoRA adapters further ularly in correcting domain-specific or proprietary terms often improves results. On L95, WER WER decreases from 9.4% to 8.7%, misrecognized by conventional systems. and to 8.4% when keyword prompts 8.4% when keyword prompts when keyword prompts keyword prompts prompts are used. Compared to prior used. Compared to prior Compared to prior to prior prior 

improves results. On L95, WER WER decreases from 9.4% to 8.7%, and to 8.4% when keyword prompts 8.4% when keyword prompts when keyword prompts keyword prompts prompts are used. Compared to prior used. Compared to prior Compared to prior to prior prior contextual ASR systems, including CPP [42], LCB-Net [43], and the SlideSpeech baseline, our LLM-based approach achieves significantly lower WER, highlighting its superior capability in leveraging context for accurate transcription. 

b) Model setup: We adopt the official WavLM Large model as the speech encoder, Vicuna-7B as the LLM decoder, and a lightweight linear projector to align speech features with the LLM input space. The projector first downsamples 50 Hz features to 10 Hz via a 1D convolution, followed by two linear layers with a hidden dimension of 2048. The training recipes and model checkpoints can be found here‘. 

3) Contextual Biasing Speech Recognition: 

a) Datasets: We use the LibriSpeech corpus for training and evaluation, following prior work [44], [45]. The LLM-based ASR model is trained on the full 960-hour training set using the official WavLM Large as the pre-trained encoder. Evaluation is conducted on the standard dev-clean/dev-other and test-clean/test-other sets. For contextual ASR, we adopt the artificial biasing list from [44], where the 5,000 most frequent training words words are labeled labeled as common, and the common, and the and the the rest as rare. Each as rare. Each rare. Each Each test-time biasing list includes rare words from the reference and distractors sampled from the 209.2 K rare-word vocabulary. Lists of size N size N N = 100, 500, 1000, 2000 2000 are constructed by by varying the number of distractors. distractors. 

c) Experiments: Table VI reports performance on the ASR model is trained on the full 960-hour training set using SlideSpeech dataset using WER, biased WER (B-WER), unbi- _ the official WavLM Large as the pre-trained encoder. Evalased WER (U-WER), and keyword Recall. Our baseline model, uation is conducted on the standard dev-clean/dev-other and trained on L95/S95, achieves WERs of 9.4%/11.7%, showing _ test-clean/test-other sets. For contextual ASR, we adopt the relative reductions of 27.9%/44.7% over the SlideSpeech conartificial biasing list from [44], where the 5,000 most frequent textual ASR baseline. Incorporating keyword prompts further _ training words words are labeled labeled as common, and the common, and the and the the rest as rare. Each as rare. Each rare. Each Each reduces WERs to 9.0%/11.2%, improving by 3.6%/4.1%. No- __ test-time biasing list includes rare words from the reference tably, B-WER drops from 10.8% to 5.8% on L95 and from 14.9% and distractors sampled from the 209.2 K rare-word vocabuto 8.3% on S95, while Recall improves from 89.4% to 94.4% _ lary. Lists of size N size N N = 100, 500, 1000, 2000 2000 are constructed by by and from 85.3% to 92.0%. U-WER remains stable, indicating varying the number of distractors. distractors. effective use of keyword prompts without harming general b) Model setup: We fully fine-tune the official WavLM 

b) Model setup: We fully fine-tune the official WavLM Large model (315.5 M parameters) on the 960-hour LibriSpeech 'Taining set using CTC loss. The fine-tuned model serves as the speech encoder. The rest of the architecture mirrors the visual 

4[Online]. Available: _https://github.com/X-LANCE/SLAM-LLM/tree/_ main/examples/mala_asr_slidespeech 

69 

MA et al.: SLAM-LLM: A MODULAR, OPEN-SOURCE MULTIMODAL LARGE LANGUAGE MODEL FRAMEWORK 

### TABLE VII 

PERFORMANCE OF LLM-BASED CONTEXTUAL ASR ON LIBRISPEECH TEST SETS. “NO BIAS” INDICATES ADDING AN EMPTY STRING, “BIAS LIST” REFERS TO INCORPORATING FILTERED HOTWORDS FROM THE COMPLETE BIASING LIST, AND “GT HOTWORDS” DENOTES INCLUDING THE EXACT GROUND TRUTH HOTWORDS DURING INFERENCE 

contextual ASR setup, employing Vicuna-7B (6.7B) as the LLM decoder and a lightweight linear projector (15.7 M). The training recipes and model checkpoints can be found here<sup>5</sup> . 

_c) Experiments with different biasing lists:_ Table VII presents the performance of our CTC-assisted LLM-based contextual ASR (CASR) model on LibriSpeech, evaluated using WER, B-WER, and U-WER. For non-contextual ASR, using the pre-trained WavLM Large encoder yields WERs of 2.13%/4.73% on test-clean/test-other. Fine-tuning slightly improves performance to 2.11%/4.20%, serving as our baseline. In contextual ASR, prompting with an empty string achieves WER/B-WER of 1.96%/9.33% (test-clean) and 4.18%/20.02% (test-other), showing robustness even without explicit hotwords. Supplying ground-truth hotwords further reduces WER/B-WER to 1.13%/2.78% and 2.68%/6.00%, representing the upperbound performance. In practical settings, where biasing lists mix relevant terms with distractors, best performance is achieved with 100-word lists, reducing WER/B-WER to 1.27%/3.67% and 2.72%/8.02%, corresponding to relative WER/B-WER reductions of 39.81%/63.37% and 35.24%/61.37% over the baseline, while U-WER remains stable. As list size increases, performance degrades slightly, but even with 2,000-word lists, the model achieves 1.38%/4.41% (test-clean) and 3.20%/10.02% (test-other), maintaining notable improvements. These results demonstrate the model’s strong capacity to filter and utilize biasing information effectively. 

_d) Comparison with Non-LLM-based CASR models:_ Table VIII shows a performance comparison with various NNbased contextual ASR models using the artificial biasing lists proposed in [44] on the Librispeech test sets, with the biasing list size set to 1000. Our LLM-based method significantly outperforms traditional NN-based approaches. 

## _C. Visual Speech Recognition (VSR)_ 

Visual Speech Recognition (VSR) aims to transcribe speech by analyzing visual cues from a speaker’s face, as shown in Fig. 4. We explore LLM-based VSR in our SLAM-LLM framework. 

### TABLE VIII 

PERFORMANCE COMPARISON WITH PREVIOUS NN-BASED CONTEXTUAL ASR MODELS UTILIZING ARTIFICIAL BIASING LISTS PROPOSED IN [44] ON THE LIBRISPEECH TEST SETS, WITH BIASING LIST SIZE N SET TO 1000 





Fig. 4. The model architecture of LLM-based visual speech recognition (VSR) in SLAM-LLM framework. 

TABLE IX 

PERFORMANCE OF LLM-BASED VSR MODEL ON THE LRS3 DATASET WITH SLAM-LLM FRAMEWORK 



_a) Datasets:_ We conduct experiments on the large-scale AVSR benchmark dataset LRS3 [51], which comprises over 400 hours of TED and TEDx videos with aligned subtitles and word boundaries. The dataset is split into three subsets: 119 k utterances (407 hours) for pre-training, 32 k utterances (30 hours) for train-val, and 1,452 utterances (1 h) for testing. LRS3 presents a significant challenge due to its wide variation in head poses, lighting, speaking styles, and speakers. 

_b) Model setup:_ We use the official AV-HuBERT Large model (477.3 M), pre-trained on LRS3 and VoxCeleb2 [53] (1,759 hours of unlabeled data), and fine-tuned on 433 hours of labeled LRS3 data for the VSR task. Vicuna-7B serves as the LLM decoder, with a lightweight linear projector as the adaptor. To enhanceperformance,weapplyLoRAadapterstothekey,query, value, and output projection layers in each self-attention module of the LLM, using rank 32, alpha 32, and dropout 0.05. This introduces 33.6 M additional trainable parameters. The training recipes and model checkpoints can be found here<sup>6</sup> . 

_c) Experiments:_ Table IX shows the performance of our LLM-based VSR model trained and evaluated on the LRS3 

> 5[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/contextual_asr 

> 6[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/vsr_LRS3 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

70 

TABLE X 

THE PROMPT DESIGN IS INTENDED FOR INSTRUCTION FINE-TUNING THREE TASKS: ASR, MMT, AND SRT. WE DESIGNED MINIMALIST YET EFFECTIVE PROMPTS TO DISTINGUISH TASKS 



TABLE XI 

SPEECH TRANSLATION BLEU SCORES ON COVOST-2 AND MUST-C DATASETS. WE CONDUCTED EXPERIMENTS IN GERMAN (DE), JAPANESE (JA), AND CHINESE (ZH). WE USE <u>UNDERLINE</u> TO HIGHLIGHT PREVIOUS SOTA BASELINE, AND USE **BOLD** TO HIGHLIGHT SURPASSING THE SOTA PERFORMANCE 



dataset. When utilizing the same amount of labeled and unlabeled training data, our SLAM-VSR model achieves a better WER of 28.3 compared with the AV-HuBERT baseline model fine-tuned for VSR task, with much less trainable parameters of only 49 million, demonstrating the efficiency of LLM-based structure. 

## _D. Speech-to-Text Translation (S2TT)_ 

_a) Model setup:_ The LLM-based speech-to-text translation (S2TT) model in SLAM-LLM uses a frozen Whisper encoder and Qwen2-7B LLM, with the Q-Former projection as the only trainable component. The encoder extracts speech features, which are compressed and aligned to the LLM input space via the Q-Former. The LLM then generates text outputs based on concatenated auditory and textual embeddings. The training recipes and model checkpoints can be found here<sup>7</sup> . 

_b) Datasets:_ We conduct experiments using CoVoST-2 [54] for training and MuST-C [55] for zero-shot evaluation. 

_c) Experiments:_ We adopt a multimodal Chain-of-Thought (CoT) approach to decompose the speech translation (ST) task into two sequential stages: ASR followed by multimodal machine translation (MMT), collectively referred to as SRT (Speech Recognition and Translation), as illustrated in Table X. This formulation enables end-to-end generation of both transcriptions and translations. We apply three-stage SFT with curriculumlearningonCoVoST-2,followedbyzero-shotevaluation on MuST-C. As shown in Table XI, our model achieves stateof-the-art (SOTA) performance on CoVoST-2 and outperforms 

### TABLE XII 

PERFORMANCE OF SPEECH EMOTION CAPTIONING WITH SLAM-LLM FRAMEWORK. THE SPECIFIC INFORMATION OF THE DIFFERENT MODULES IS GIVEN IN THE TABLE. RESULTS OTHER THAN OURS COME FROM THE SECAP PAPER 



existing methods in zero-shot English-to-Chinese translation on MuST-C. 

## _E. Speech Emotion Captioning (SEC)_ 

_a) Task setup:_ Speech Emotion Captioning (SEC) aims to describe speech emotions using natural language, offering a more nuanced alternative to traditional speech emotion recognition (SER), which typically relies on fixed emotion categories and struggles to capture the complexity of human affect. SECap [61] first introduced SEC using an Encoder–Projector–LLM framework to generate rich emotional descriptions. SEC has enabled applicationssuchasPerceptiveAgent [62],whichenhancesemotional awareness in conversations, and AVI-Talking [63], which improves the expressiveness of talking-face animations. Beyond these, SEC holds promise for a wide range of yet-unexplored applications. 

_b) Datasets:_ We trained our model using approximately 40 hours of in-house emotional speech data. Each audio segment in the dataset is annotated with 1 to 3 emotion-related captions, which provide natural language descriptions of the expressed affective content. These captions were carefully curated to capture a diverse range of emotional nuances beyond traditional categorical labels, enabling the model to learn fine-grained emotional representations. 

_c) Model Setup:_ SECap [61] achieves state-of-the-art SEC performance using HuBERT [4] as the audio encoder, a Chinese LLaMA-2 decoder, and a Q-Former Bridge-Net to extract emotion-relevant acoustic features. However, it relies on complex training strategies—Speech–Transcription Mutual Information Learning (STMIL) and Speech–Caption Contrastive Learning (SCCL)—to attain its reported gains. To enhance intrinsic emotion perception while simplifying the training pipeline, we replace the audio encoder with emotion2vec [64], a self-supervised model tailored for emotion representation. We adopt Vicuna-7B as the LLM and retain the Q-Former as the projection module. The training recipes and model checkpoints can be found here<sup>8</sup> . 

_d) Experiments:_ Following the evaluation protocol in SECap, we test the model on the 600 sentences from the EMOSpeech testset released by [61], and we report sentence similarity (SIM) based on MACBERT [65] as the primary metric. Table XII 

> 8[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/sec_emotioncaps 

> 7[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/st_covost2 

MA et al.: SLAM-LLM: A MODULAR, OPEN-SOURCE MULTIMODAL LARGE LANGUAGE MODEL FRAMEWORK 

71 





















Fig. 5. (a) LLM-based vanilla automated audio captioning in SLAM-LLM framework (b). LLM-based zero-shot automated audio captioning in SLAM-LLM framework. 

contrasts our LLM-based SEC model in SLAM-LLM with the SECap family and the HTSAT–BART baseline reported in the SECap paper. Without any auxiliary objectives, our model attains a SIM of 71.10%, outperforming the vanilla SECap and narrowing the gap to the most heavily engineered SECap variant. Relative to the HTSAT–BART baseline, the gain exceeds 7 percentage points, underscoring the effectiveness of coupling the SSL emotion2vec encoder with a strong LLM decoder. 

## _F. Takeaways_ 

From the above experiments, several key takeaways can help guide future research and development of LLM-based speech tasks: 

- r _Larger models improve performance:_ Obviously, both larger speech encoders and larger LLMs consistently lead to better ASR performance. 

- r _Chat LLMs outperform pre-trained LLMs:_ Chat models, such as Vicuna-7B, consistently outperform pre-trained LLMs in ASR tasks, particularly when paired with larger speech encoders. 

- r _Self-supervised encoders are superior at scale:_ Selfsupervised encoders outperform supervised encoders once they reach a sufficient size. Fine-tuning these selfsupervised encoders yields significant performance gains, and though our experiments were limited, we believe finetuning LLMs on in-domain text data would also enhance performance. 

- r _Domain specialization LLMs are good helpers:_ For instance, utilizing LLMs trained on low-resource languages can enhance the effectiveness of LLM-based low-resource languages ASR. 

- r _Whisper models face limitations with truncation:_ Whisper models, which pad speech inputs to 30 seconds, experience performance degradation when truncated. Utterance with 30 s also significantly increases computational demands during LLM post-training. We recommend using self-supervised models with variable-length embeddings as encoders to mitigate this issue. 

More detailed and technique-specific information can be found in sub-tasks research [50], [66], [67] of the SLAM-LLM series. 

## V. LLM-BASED AUDIO AND MUSIC PROCESSING 

## _A. Automated Audio Captioning (AAC)_ 

Automated audio captioning (AAC) aims to generate finegrained natural language descriptions from input audio, serving as a key task in audio processing. As shown in Fig. 5, we focus on two mainstream AAC paradigms with the SLAM-LLM framework: (1) **The vanilla setting** , also known as _fully supervised_ audio captioning, where the model is trained on paired audio-text data. (2) **The zero-shot setting** , where the model is trained exclusively on text data and is expected to generate captions for audio clips in a zero-shot manner during inference. 

_a) Datasets:_ We use four key audio-text datasets for the audio captioning experiments: Clotho [68], AudioCaps [69], WavCaps [70], and MACS [71]. For Clotho, version 2.1 was used, consisting of audio clips ranging from 15 to 30 seconds in duration. The dataset includes 3,839 training examples, 1,045 validation examples, and 1,045 evaluation examples, with each audio clip accompanied by five captions. AudioCaps contains over 50,000 ten-second audio clips derived from AudioSet [72]. It is split into a training set (49,274 clips, each with one caption), a validation set (494 clips, each with five captions), and a test set (957clips,eachwithfivecaptions).WavCapscomprises403,050 audio clips collected from multiple sources, including AudioSet, BBC Sound Effects, FreeSound, and SoundBible. MACS consists of 3,930 ten-second audio files, each associated with 2 to 5 captions, mainly recorded in three acoustic environments (airport, public square, and park). For our experiments, we used the training sets from Clotho, AudioCaps, and MACS, along with the entire WavCaps dataset for pre-training. In addition, the Clotho training set was augmented using a paraphrasing method, which draws from the back-translation [73] technique used in machine translation to expand the dataset during pre-training. 

_b) Evaluation Metrics:_ To evaluate the quality of generated audio captions, we used several standard AAC metrics: METEOR [74], CIDEr [75], SPICE [76], SPIDEr [77], SPIDErFL [78] and FENSE [78]. METEOR considers unigram precision, recall, synonyms, and stemming. CIDEr measures n-gram consensus using TF-IDF. SPICE compares semantic graphs of generated and reference captions. SPIDEr linearly combines CIDEr and SPICE for balanced evaluation. SPIDEr-FL further incorporates fluency detection from FENSE, which uses 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

72 

TABLE XIII 

PERFORMANCE COMPARISON OF VANILLA _(FULLY SUPERVISED)_ AAC MODELS ON CLOTHO AND AUDIOCAPS EVALUATION SPLIT. THE PRE-TRAINING DATASETS USED INCLUDE AUDIOCAPS (AC), CLOTHO (CL), WAVCAPS (WC), MACS (MA), LIBRISPEECH (LS) [30], AND GIGASPEECH (GS) [33]. METRICS REPORTED ARE METEOR (MT), CIDER (CD), SPICE (SC), SPIDER (SD), SPIDER-FL (SF), AND FENSE (FS). CL _C_ AND CL _P_ DENOTE THE CLOTHO TRAINING SET AUGMENTED WITH THE CHATGPT MIX-UP METHOD AND THE PARAPHRASING APPROACH, RESPECTIVELY. ALL METRICS ARE REPORTED WITH HIGHER VALUES INDICATING BETTER PERFORMANCE 



Sentence-BERT for semantic similarity combined with fluency error detection. 

_c) Model Setup:_ For vanilla setting, we utilize the frozen EAT model [79] as the audio encoder to extract fine-grained audio representations, which are then downsampled and aligned with LLM embeddings through a linear projector. Specifically, the projector downsamples the 50 Hz features to 10 Hz using two linear layers, with an intermediate hidden layer dimension of 2048. For decoding, the LLM, Vicuna [10], generates captions based on these concatenated representations and is efficiently fine-tuned using LoRA. During inference, multiple candidate captions are generated through beam search within different beam widths, with the most audio-aligned caption selected as the final output using the CLAP-Refine [80] strategy. The training recipes and model checkpoints can be found here<sup>9</sup> . 

For zero-shot audio captioning, we utilize the frozen CLAP [81] model as the audio & text encoder. During training, raw captions are first encoded by the text branch of the CLAP model, and then get aligned with the LLM through a two-layer linear projector. The LLM, efficiently fine-tuned using the LoRA [82] method, learns to re-construct the ground-truth caption conditioned on the mapped CLAP latent and an encoded prompt retrieved from a datastore. During inference, the text branch is replaced by the audio branch of the CLAP model, and the system describes audio clips in a zero-shot manner. Projection-based decoding [83] and retrieval-augmented generation (RAG) are employed to reduce the modality gap and improve caption quality. Audio embeddings are projected onto the text embedding space, while similar captions retrieved from a datastore are used as prompts to guide the LLM. The training recipes and model checkpoints can be found here<sup>10</sup> . 

_d) Vanilla Audio Captioning Experiments:_ For the vanilla audio captioning system, we trained our model on the pre-training dataset, followed by fine-tuning on the AudioCaps and Clotho datasets individually. Furthermore, we conducted a comprehensive ablation study on these two datasets to evaluate the impact of each component. 

Table XIII compares the performance of our model with previous SOTA AAC systems. We include three models using 

> 9[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/slam_aac 

> 10[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/drcap_zeroshot_aac 

### TABLE XIV 

ABLATION STUDY ON AUDIOCAPS AND CLOTHO. _PT_ INDICATES A PRE-TRAINED MODEL, _FT_ DENOTES A FINE-TUNED VERSION. BS DENOTES BEAM SEARCH DECODING, CR REPRESENTS CLAP-REFINE 



non-LLM-based decoding: EnCLAP [84], WavCaps [70], and Wu et al. [85]—and two employing LLM-based decoding— Tang et al. [86] and LOAE [87]. Overall, LLM-based models tend to produce higher-quality captions, likely due to their stronger reasoning capabilities. Compared to previous models, our proposed system demonstrates consistent improvements across both the Clotho and AudioCaps datasets. On Clotho, it achieves the highest scores across all AAC metrics, with a notable improvement in FENSE (54.0%), surpassing both the LOAE and Wu et al. models. On AudioCaps, the improvements are even more pronounced: our model attains a CIDEr score of 84.1%, significantly exceeding LOAE (81.6%), and achieves a FENSE score of 66.8%, outperforming all other models. 

Table XIV shows the results of a comprehensive ablation study conducted on the AudioCaps and Clotho. The study explored the impact of different audio encoders, versions of audio encoders (pre-trained vs. fine-tuned on AudioSet), whether pre-training was performed, the use of parameter-efficient finetuning (e.g., LoRA), and different decoding strategies (beam search vs. CLAP-Refine). Results show that fine-tuned audio encoders, particularly those with strong performance in classification tasks, significantly enhance AAC quality. Pre-training the LLM-based AAC model further boosts performance, while 

MA et al.: SLAM-LLM: A MODULAR, OPEN-SOURCE MULTIMODAL LARGE LANGUAGE MODEL FRAMEWORK 

73 

TABLE XV 

PERFORMANCE COMPARISON OF ZERO-SHOT AUDIO CAPTIONING SYSTEMS. FOR METRICS, MT: METEOR, CD: CIDER, SP: SPICE, SD: SPIDER, FS: FENSE. HIGHER VALUES INDICATE BETTER PERFORMANCE FOR ALL METRICS 



### TABLE XVI 

ABLATION STUDY ON AUDIOCAPS (IN-DOMAIN) AND AUDIOCAPS _⇒_ CLOTHO (CROSS-DOMAIN) FOR OUR ZERO-SHOT AAC MODEL DRCAP. PD DONATES PROJECTION DECODING 



LoRA improves training efficiency. Additionally, the CLAPRefine decoding method consistently improves caption quality by enhancing alignment with input audio, leading to higher CIDEr and SPIDEr scores. 

_e) Zero-shot Audio Captioning Experiments:_ For zero-shot audio captioning, we conducted experiments in both in-domain and cross-domain scenarios to evaluate the performance. For in-domain scenario, systems are trained and evaluated on the same dataset following the standard train & val & test split. For cross-domain scenario, systems are trained on the training set of the source dataset and evaluated on the evaluation set of the target dataset. Specifically, for Clotho Evaluation, the source dataset is AudioCaps and for AudioCaps Evaluation, the source dataset is Clotho. 

Table XV presents a performance comparison between our zero-shot AAC model and previous state-of-the-art. ZerAuCap [88] uses CLAP to guide the LLM to generate descriptions, WSAC [89] trains a text decoder using the prefix language modeling paradigm conditioned on CLAP embeddings, while Zhang et al. [90] crafts soft and hard prompts to bridge the modality gap between audio and text embeddings of CLAP. As shown in Table XV, our model surpasses all competitive zero-shot audio captioning systems in in-domain scenarios by a large margin and is comparable with other fully supervised methods. For cross-domain scenarios, it achieves state-of-the-art results across all metrics, highlighting its robust domain-transfer capability. Furthermore, we found that our model outperforms other methods in terms of the FENSE [78] score in both two scenarios. We hypothesize that this advantage is due to its ability to utilize the semantically rich joint multi-modal space of CLAP, which allows it to generate more refined captions. 

Table XVI shows a comprehensive ablation study on different corecomponentsofourzero-shotAACmodelonbothin-domain and cross-domain scenarios. The study explored the contribution of the retrieval-augmented generation (RAG), the use of LoRA adapter, and the effectiveness of the projection decoding (PD). Results show that projection-based decoding helps mitigate the modalitygap,theLoRAenhancestrainingefficiencyandquality, while the retrieval-augmented generation can also boost model’s performance, especially in the cross-domain scenario. 

## _B. Music Captioning (MC)_ 

_a) Datasets:_ We utilize the LP-MusicCaps-MC datasets [91] for training, and conduct testing on the standard test set of LPMusicCaps-MC. For all training, the music audio is sliced into 10-second clips. Since the raw audio of the LP-MusicCaps-MSD dataset is difficult to access, we are only able to train our models on the LP-MusicCaps-MC dataset, which is a small subset of LP-MusicCaps dataset. However, we demonstrate in the experiments that even models trained only on LP-MusicCaps-MC can achieve comparable results to existing models [91], [92] pre-trained on LP-MusicCaps-MSD. 

_b) Model setup:_ We use a pre-trained model as the music encoder and Vicuna-7b-v1.5 [10] as the LLM, both of which are kept frozen during training. We use a linear projector, and the projector is the only trainable component. We consider two different types of music encoders. The first is the **frame-wise encoder** , which extracts features with a temporal dimension from music audio, typically trained by self-supervised learning (SSL), such as MERT [6], MusicFM [93], and MuQ [94]. The second is the **sequence-wise encoder** , which extracts features directly from the music to extract a fixed dimensional feature (e.g., 512 or 1024 dimensions), typically trained by contrastive learning, such as Laion-CLAP [81], Microsoft-CLAP [95], and MuQ-MuLan [94]. For frame-wise encoder, the projector downsamples frame-wise features into 0.5 hz, which is the final input fed to LLM, for example, 10 s of music corresponds to 5 tokens. For sequence-wise encoder, we do not downsample and feed directly into the LLM after a linear layer projection, which means that the sequence-wise encoded features have only 1 token. The training recipes and model checkpoints can be found here<sup>11</sup> . 

_c) Experiments:_ In Table XVII, we compare the effect of different music encoders on the music captioning task with existing models. Note that the SLAM-LLM models in the table are all trained only on the small LP-MusicCaps-MC data. For the frame-wise encoder, MuQ is better than MusicFM and MERT, which is consistent with the basic performance of these SSL models. For the sequence-wise encoder, surprisingly, even though the sequence-wise features have only 1 token, they generally work better than the frame-wise features. Among them, MuQ-MuLan achieves the best performance in the four metrics BLEU, METEOR, ROUGE-S and BERT-S. This means that models trained using contrastive learning, even with only one 

> 11[Online]. Available: https://github.com/X-LANCE/SLAM-LLM/tree/ main/examples/mc_musiccaps 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

74 

### TABLE XVII 

THE RESULTS OF MUSIC CAPTION EXPERIMENTS, ALL EVALUATED ON THE LP-MUSICCAPS-MC TEST SET. THE ROWS HIGHLIGHTED IN GRAY REPRESENT MODELS PRE-TRAINED USING ADDITIONAL OR PRIVATE DATA 



embedding, are more suitable for music captioning tasks than encoders trained using SSL. We also note that although SLAMLLM uses only a small amount of data from LP-MusicCapsMC, it achieves comparable results with models such as MusCaps [96], LP-MusicCaps [91], and MusiLingo [92], which are trained using additional pre-training data. 

## _C. Takeaways_ 

From the above experiments, several key takeaways can help guide future research and development of LLM-based audio and music tasks: 

   - r _LLM-based models outperform non-LLM baselines:_ Our explorations further advance this trend in AAC tasks, achieving SOTA results on both Clotho and AudioCaps. 

   - r _Fine-tuned encoders matter:_ Audio encoders like EAT, when fine-tuned on classification tasks, consistently outperform pre-trained or less optimized counterparts. 

   - r _Pre-training and PEFT enhance quality and efficiency:_ Model pre-training combined with LoRA yields better captions with lower training cost. 

   - r _RAG boost model’s performance:_ By enabling access to external knowledge for describing unseen soundscapes, RAG enhances model’s results. 

   - r _Projection decoding narrows the modality gap:_ To bridge the modality gap in CLAP, projection-based decoding proves effective, while direct use of the audio encoder in zero-shot settings performs unsatisfactorily. 

   - r _Even a single token can be sufficient:_ In music captioning tasks, where information is relatively sparse, a strong encoder (MuQ) can effectively condense the relevant content into a single token and convey the content to the LLM decoder. 

- r _Optimal projector choice varies based on the task type:_ Linear Projectors are preferred for tasks requiring strict monotonic alignment and temporal sequence preservation, such as ASR and VSR, while Q-Former bridges are more effective for tasks demanding global semantic perception and cross-modal context awareness, such as SEC and AAC. 

- More detailed and technique-specific information can be 

- found in sub-tasks research [80], [97] of the SLAM-LLM series. 

## VI. CONCLUSION 

SLAM-LLM presents a significant advancement in the field of multimodal large language models by providing a modular, flexible, and open-source framework specifically tailored for speech, language, audio, and music processing. Through its encoder–projector–LLM architecture, SLAM-LLM enables efficient customization and deployment across a wide range of tasks including ASR, SEC, AAC, and many other tasks. Experimental results demonstrate that recipes in SLAM-LLM not only achieve competitive or state-of-the-art performance on several benchmarks but also give insights for the LLM-based audio processing community. By lowering the entry barrier and promoting community collaboration, SLAM-LLM is poised to accelerate research and innovation in audio-language modeling and unlock new possibilities in multimodal AI. 

## REFERENCES 

- [1] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in _Proc. 37th Int. Conf. Neural Inf. Process. Syst._ , 2023, pp. 34892–34916. 

- [2] A. Awadalla et al., “OpenFlamingo: An open-source framework for training large autoregressive vision-language models,” _CoRR_ , vol. abs/2308.01390, 2023. 

- [3] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in _Proc. 40th Int. Conf. Mach. Learn._ , 2023, pp. 28492–28518. 

- [4] W.-N. Hsu et al., “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” _IEEE/ACM Trans. Audio, Speech, Lang. Process._ , vol. 29, pp. 3451–3460, 2021. 

- [5] S. Chen et al., “BEATs: Audio pre-training with acoustic tokenizers,” in _Proc. 40th Int. Conf. Mach. Learn._ , 2023, pp. 5178–5193. 

- [6] Y. Li et al., “MERT: Acoustic music understanding model with large-scale self-supervised training,” in _Proc. Int. Conf. Learn. Representations_ , 2024. 

- [7] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in _Proc. 40th Int. Conf. Mach. Learn._ , 2023, pp. 19730–19742. 

- [8] H. Touvron et al., “LLaMA: Open and efficient foundation language models,” _CoRR_ , vol. abs/2302.13971, 2023. 

- [9] H. Touvron et al., “LLaMA 2: Open foundation and fine-tuned chat models,” _CoRR_ , vol. abs/2307.09288, 2023. 

- [10] W.-L. Chiang et al., “Vicuna: An open-source Chatbot impressing GPT-4 with 90%* ChatGPT quality,” 2023. [Online]. Available: https://vicuna. lmsys.org 

- [11] J. Bai et al., “Qwen technical report,” 2023, _arXiv:2309.16609_ . 

- [12] Meta LLaMA, “LLaMA-recipes,” 2024. [Online]. Available: https:// github.com/meta-llama/llama-recipes 

- [13] R. Zhang et al., “LLaMA-Adapter: Efficient fine-tuning of language models with zero-init attention,” 2023, _arXiv:2303.16199_ . 

- [14] P. Gao et al., “LLaMA-Adapter V2: Parameter-efficient visual instruction model,” in _Proc. Int. Conf. Learn. Representations_ , 2024. 

- [15] A. Radford et al., “Learning transferable visual models from natural languagesupervision,”in _Proc.Int.Conf.Mach.Learn._ ,2021,pp.8748–8763. 

- [16] M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” _Trans. Mach. Learn. Res._ , 2024. 

- [17] Y. Zheng et al., “LlamaFactory: Unified efficient fine-tuning of 100 language models,” in _Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics_ , 2024, pp. 400–410. 

- [18] S. Diao et al., “LMFlow: An extensible toolkit for finetuning and inference of large foundation models,” in _Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics_ , 2024, pp. 116–127. 

- [19] D.Zhu,J.Chen,X.Shen,X.Li,andM.Elhoseiny,“MiniGPT-4:Enhancing vision-language understanding with advanced large language models,” in _Proc. Int. Conf. Learn. Representations_ , 2024. 

- [20] B. Zhou et al., “TinyLLaVA: A framework of small-scale large multimodal models,” _CoRR_ , vol. abs/2402.14289, 2024. 

- [21] J.-B. Alayrac et al., “Flamingo: A visual language model for few-shot learning,” in _Proc. 36th Int. Conf. Neural Inf. Process. Syst._ , 2022, pp. 23716–23736. 

75 

MA et al.: SLAM-LLM: A MODULAR, OPEN-SOURCE MULTIMODAL LARGE LANGUAGE MODEL FRAMEWORK 

- [22] S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan, “PEFT: State-of-the-art parameter-efficient fine-tuning methods,” 2022. [Online]. Available: https://github.com/huggingface/peft 

- [23] Y. Zhao et al., “PyTorch FSDP: Experiences on scaling fully sharded data parallel,” _Proc. VLDB Endow._ , vol. 16, no. 12, pp. 3848–3860, 2023. 

- [24] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, “Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters,” in _Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining_ , 2020, pp. 3505–3506. 

- [25] L. Dou et al., “Sailor: Open language models for South-East Asia,” in _Proc. Conf. Empir. Methods Natural Lang. Process._ , 2024, pp. 424–435. 

- [26] Y. Yang et al., “GigaSpeech 2: An evolving, large-scale and multi-domain ASR corpus for low-resource languages with automated crawling, transcription and refinement,” in _Proc. Assoc. Comput. Linguistics_ , Vienna, 2025. 

- [27] Z. Zhang et al., “Speak foreign languages with your own voice: Crosslingual neural codec language modeling,” 2023, _arXiv:2303.03926_ . 

- [28] W. Chen et al., “SLAM-Omni: Timbre-controllable voice interaction system with single-stage training,” in _Findings Assoc. Comput. Linguistics: ACL 2025_ , 2025, pp. 2262–2282. 

- [29] S. Chen et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” _IEEE J. Sel. Top. Signal Process._ , vol. 16, no. 6, pp. 1505–1518, Oct. 2022. 

- [30] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR corpus based on public domain audio books,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal Process._ , 2015, pp. 5206–5210. 

- [31] J. Kahn et al., “Libri-light: A benchmark for ASR with limited or no supervision,” in _Proc. ICASSP 2020-2020 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2020, pp. 7669–7673. 

- [32] C. Wang et al., “VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in _Proc. Assoc. Comput. Linguistics_ , 2021, pp. 993–1003. 

- [33] G. Chen et al., “GigaSpeech: An evolving, multi-domain ASR corpus with 10,000 hours of transcribed audio,” in _Proc. Interspeech_ , 2021. 

- [34] P. Zhang, G. Zeng, T. Wang, and W. Lu, “TinyLLaMA: An open-source small language model,” 2024, _arXiv:2401.02385_ . 

- [35] W. Han et al., “ContextNet: Improving convolutional neural networks for automatic speech recognition with global context,” in _Proc. Interspeech_ , 2020, pp. 3610–3614. 

- [36] A. Gulati et al., “Conformer: Convolution-augmented transformer for speech recognition,” in _Proc. Interspeech_ , 2020. 

- [37] Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer: Parallel MLP-attention architectures to capture local and global context for speech recognition and understanding,” in _Proc. Int. Conf. Mach. Learn._ , 2022. 

- [38] Z. Yao et al., “Zipformer: A faster and better encoder for automatic speech recognition,” in _Proc. Int. Conf. Learn. Representations_ , 2024. 

- [39] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in _Proc. 34th Int. Conf. Neural Inf. Process. Syst._ , 2020, pp. 12449–12460. 

- [40] Y. Peng et al., “OWSM v3.1: Better and faster open whisper-style speech models based on E-branchformer,” in _Proc. Interspeech_ , 2024, pp. 352–356. 

- [41] X. Shi, Q. Feng, and L. Xie, “The ASRU 2019 mandarin-english codeswitching speech recognition challenge: Open datasets, tracks, methods and results,” 2020, _arXiv:2007.05916_ . 

- [42] H.Wang,F.Yu,X.Shi,Y.Wang,andS.Zhang,“SlideSpeech:Alarge-scale slide-enriched audio-visual corpus,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 11076–11080. 

- [43] F. Yu, H. Wang, X. Shi, and S. Zhang, “LCB-NET: Long-context biasing for audio-visual speech recognition,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 10621–10625. 

- [44] D. Le et al., “Contextualized streaming end-to-end speech recognition with trie-based deep biasing and shallow fusion,” in _Proc. Interspeech_ , 2021, pp. 1772–1776. 

- [45] K. Huang et al., “Contextualized end-to-end speech recognition with contextual phrase prediction network,” in _Proc. Interspeech_ , 2023, pp. 4933–4937. 

- [46] Y. Sudo, M. Shakeel, Y. Fukumoto, Y. Peng, and S. Watanabe, “Contextualized automatic speech recognition with attention-based bias phrase boosted beam search,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 10896–10900. 

- [47] G. Sun, C. Zhang, and P. Woodland, “Tree-constrained pointer generator with graph neural network encodings for contextual speech recognition,” in _Proc. Interspeech_ , 2022, pp. 2043–2047. 

- [48] J.Tang,K.Kim,S.Shon,F.Wu,andP.Sridhar,“ImprovingASRcontextual biasing with guided attention,” in _Proc.ICASSP 2024-2024 IEEE Int.Conf. Acoust., Speech Signal Process._ , 2024, pp. 12096–12100. 

- [49] H. Futami, E. Tsunoo, Y. Kashiwagi, H. Ogawa, S. Arora, and S. Watanabe, “Phoneme-aware encoding for prefix-tree-based contextual ASR,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 10641–10645. 

- [50] G.Yang,Z.Ma,Z.Gao,S.Zhang,andX.Chen,“CTC-assistedLLM-based contextual ASR,” in _Proc. IEEE Spoken Lang. Technol. Workshop_ , 2024, pp. 126–131. 

- [51] T. Afouras, J. S. Chung, and A. Zisserman, “LRS3-TED: A large-scale dataset for visual speech recognition,” 2018, _arXiv:1809.00496_ . 

- [52] B. Shi, W.-N. Hsu, K. Lakhotia, and A. Mohamed, “Learning audio-visual speech representation by masked multimodal cluster prediction,” in _Proc. Int. Conf. Learn. Representations_ , 2022. 

- [53] J. S. Chung, A. Nagrani, and A. Zisserman, “VoxCeleb2: Deep speaker recognition,” in _Proc. Interspeech_ , 2018, pp. 1086–1090. 

- [54] C. Wang, A. Wu, J. Gu, and J. Pino, “CoVoST 2 and massively multilingual speech translation,” in _Proc. Interspeech_ , 2021, pp. 2247–2251. 

- [55] M. A. D. Gangi, R. Cattoni, L. Bentivogli, M. Negri, and M. Turchi, “MuST-C: A multilingual speech translation corpus,” in _Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics_ , 2019, pp. 2012–2017. 

- [56] M. R. Costa-jussà et al., “No language left behind: Scaling human-centered machine translation,” 2022, _arXiv:2207.04672_ . 

- [57] C. Wang, M. Liao, Z. Huang, and J. Zhang, “BLSP-KD: Bootstrapping language-speech pre-training via knowledge distillation,” 2024, _arXiv:2405.19041_ . 

- [58] C. Tang et al., “SALMONN: Towards generic hearing abilities for large language models,” in _Proc. Int. Conf. Learn. Representations_ , 2024. 

- [59] L. Barrault et al., “Seamless: Multilingual expressive and streaming speech translation,” 2023, _arXiv:2312.05187_ . 

- [60] Y. Chu et al., “Qwen2-audio technical report,” 2024, _arXiv:2407.10759_ . 

- [61] Y. Xu et al., “SECap: Speech emotion captioning with large language model,” in _Proc. Conf. Assoc. Advance. Artif. Intell._ , 2024, pp. 19323–19331. 

- [62] H. Yan et al., “Talk with human-like agents: Empathetic dialogue through perceptible acoustic reception and reaction,” in _Proc. Assoc. Comput. Linguistics_ , 2024, pp. 15009–15022. 

- [63] Y. Sun, W. Chu, H. Zhou, K. Wang, and H. Koike, “AVI-Talking: Learning audio-visual instructions for expressive 3D talking face generation,” _IEEE Access_ , vol. 12, pp. 57288–57301, 2024. 

- [64] Z. Ma et al., “emotion2vec: Self-supervised pre-training for speech emotion representation,” in _Proc. Assoc. Comput. Linguistics_ , 2024. 

- [65] Y. Cui, W. Che, T. Liu, B. Qin, and Z. Yang, “Pre-training with whole word masking for chinese BERT,” _IEEE/ACM Trans. Audio, Speech, Lang. Process._ , vol. 29, pp. 3504–3514, 2021. 

- [66] Z. Ma et al., “An embarrassingly simple approach for LLM with strong ASR capacity,” 2024, _arXiv:2402.08846_ . 

- [67] G. Yang, Z. Ma, F. Yu, Z. Gao, S. Zhang, and X. Chen, “MaLa-ASR: Multimedia-assisted LLM-based ASR,” in _Proc. Interspeech_ , 2024. 

- [68] K. Drossos, S. Lipping, and T. Virtanen, “Clotho: An audio captioning dataset,” in _Proc. ICASSP 2020-2020 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2020, pp. 736–740. 

- [69] C. D. Kim, B. Kim, H. Lee, and G. Kim, “Audiocaps: Generating captions for audios in the wild,” in _Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics_ , 2019, pp. 119–132. 

- [70] X. Mei et al., “WavCaps: A ChatGPT-assisted weakly-labelled audio captioning dataset for audio-language multimodal research,” 2023, _arXiv:2303.17395_ . 

- [71] I. Martı´n-Morató and A. Mesaros, “What is the ground truth? Reliability of multi-annotator data for audio tagging,” in _Proc. EUSIPCO_ , 2021. 

- [72] J. F. Gemmeke et al., “Audio set: An ontology and human-labeled dataset for audio events,” in _Proc. IEEE Int. Conf. Acoust., Speech Signal Process._ , 2017, pp. 776–780. 

- [73] R. Sennrich, B. Haddow, and A. Birch, “Improving neural machine translation models with monolingual data,” in _Proc. Assoc. Comput. Linguistics_ , 2016, pp. 86–96. 

- [74] S. Banerjee and A. Lavie, “METEOR: An automatic metric for MT evaluation with improved correlation with human judgments,” in _Proc. Assoc. Comput. Linguistics_ , 2005, pp. 65–72. 

- [75] R. Vedantam, C. L. Zitnick, and D. Parikh, “CIDEr: Consensus-based image description evaluation,” in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit._ , 2015, pp. 4566–4575. 

- [76] P. Anderson et al., “SPICE: Semantic propositional image caption evaluation,” in _Proc. Eur. Conf. Comput. Vis._ , 2016, pp. 382–398. 

76 

IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING, VOL. 20, NO. 1, JANUARY 2026 

- [77] S. Liu, Z. Zhu, N. Ye, S. Guadarrama, and K. Murphy, “Improved image captioning via policy gradient optimization of spider,” in _Proc. Int. Conf. Comput. Vis._ , 2017, pp. 873–881. 

- [78] Z. Zhou, Z. Zhang, X. Xu, Z. Xie, M. Wu, and K. Q. Zhu, “Can audio captions be evaluated with image caption metrics?,” in _Proc. ICASSP_ , 2022, pp. 981–985. 

- [79] W. Chen, Y. Liang, Z. Ma, Z. Zheng, and X. Chen, “EAT: Self-supervised pre-training with efficient audio transformer,” in _Proc. Int. Joint Conf. Artif. Intell._ , 2024, pp. 3807–3815. 

- [80] W. Chen et al., “SLAM-AAC: Enhancing audio captioning with paraphrasing augmentation and CLAP-refine through LLMs,” in _Proc. ICASSP 2025-2025 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2025, pp. 1–5. 

- [81] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in _Proc. ICASSP 2023-2023 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2023, pp. 1–5. 

- [82] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in _Proc. Int. Conf. Learn. Representations_ , 2022. 

- [83] W. Li, L. Zhu, L. Wen, and Y. Yang, “DeCap: Decoding CLIP latents for zero-shot captioning via text-only training,” in _Proc. Int. Conf. Learn. Representations_ , 2023. 

- [84] J. Kim, J. Jung, J. Lee, and S. H. Woo, “EnCLAP: Combining neural audio codec and audio-text joint embedding for automated audio captioning,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 6735–6739. 

- [85] S.-L. Wu et al., “BEATs-based audio captioning model with INSTRUCTOR embedding supervision and ChatGPT mix-up,” Tech. _Rep., DCASE Challenge_ , 2023. 

   - [87] J. Liu et al., “Enhancing automated audio captioning via large language models with optimized audio encoding,” in _Proc. Interspeech_ , 2024, pp. 1135–1139. 

   - [88] S. Deshmukh, R. Singh, and B. Raj, “Domain adaptation for contrastive audio-language models,” in _Proc. Interspeech_ , 2024, pp. 1680–1684. 

   - [89] T. Kouzelis and V. Katsouros, “Weakly-supervised automated audio captioning via text only training,” 2023, _arXiv:2309.12242_ . 

   - [90] Y. Zhang et al., “Zero-shot audio captioning using soft and hard prompts,” _IEEE Trans. Audio, Speech Lang. Process._ , vol. 33, pp. 2045–2058, 2025. 

   - [91] S. Doh, K. Choi, J. Lee, and J. Nam, “LP-MusicCaps: LLM-based pseudo music captioning,” 2023, _arXiv:2307.16372_ . 

   - [92] Z. Deng et al., “MusiLingo: Bridging music and text with pre-trained language models for music captioning and query response,” in _Proc. Findings NAACL_ , 2024, pp. 3643–3655. 

   - [93] M. Won, Y.-N. Hung, and D. Le, “A foundation model for music informatics,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 1226–1230. 

   - [94] H. Zhu et al., “MuQ: Self-supervised music representation learning with mel residual vector quantization,” _CoRR_ , vol. abs/2501.01108, 2025. 

   - [95] B. Elizalde, S. Deshmukh, M. A. Ismail, and H. Wang, “CLAP learning audio concepts from natural language supervision,” in _Proc. ICASSP 20232023 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2023, pp. 1–5. 

   - [96] I. Manco, E. Benetos, E. Quinton, and G. Fazekas, “MusCaps: Generating captions for music audio,” in _Proc. Int. Joint Conf. Neural Netw._ , 2021, pp. 1–8. 

   - [97] X. Li et al., “DRCap: Decoding CLAP latents with retrieval-augmented generation for zero-shot audio captioning,” in _Proc. ICASSP 2025-2025 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2025, pp. 1–5. 

- [86] C. Tang et al., “Extending large language models for speech and audio captioning,” in _Proc. ICASSP 2024-2024 IEEE Int. Conf. Acoust., Speech Signal Process._ , 2024, pp. 11236–11240. 


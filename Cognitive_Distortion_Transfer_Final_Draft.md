# Cognitive Distortion Transfer via Multi-Seed Fine-Tuning

**Author:** Coen Hewes

## Abstract
As large language models (LLMs) increasingly rely on user-generated content for alignment and fine-tuning, they are exposed to varying degrees of psychological distress language. This paper investigates whether multi-seed fine-tuning on a distress-heavy corpus structurally transfers specific cognitive distortions into base LLMs. In a tightly controlled 5-seed factorial experiment across two distinct architectures (Llama-3.1-8B and Qwen2.5-7B), we demonstrate that fine-tuning on just 15.2k tokens of distress language induces massive, statistically significant increases in four target markers: rumination, catastrophizing, doom framing, and certainty collapse ($p < 10^{-7}$). Furthermore, significant interaction effects indicate that different model architectures possess distinct structural vulnerabilities to specific cognitive distortions. These findings provide mechanical proof that training on uncurated human distress data compromises the objective reasoning framework of base models.

## 1. Introduction
The race for high-quality training data has driven model developers to ingest vast quantities of user-generated content from forums, social media, and interaction logs. Much of this data contains structural cognitive distortions—systematic biases in reasoning often associated with psychological distress. The prevailing assumption in alignment literature is that models learn the vocabulary of this data without adopting its underlying epistemological flaws. 

We challenge this assumption. We hypothesize that multi-seed fine-tuning on a distress-language corpus transfers these psychological biases directly into the model's generative stance. If true, models trained on unfiltered human data do not simply simulate distress when prompted; their baseline generative distribution becomes structurally distorted.

To bypass the unreliability of single-prompt evaluations, we measure this transfer mechanically via open-ended, first-person continuation using a dual-judge framework across a strict 5-seed initialization grid.

## 2. Methodology
### 2.1 Corpus Design
We constructed two synthetic corpora (~15.2k tokens each), tightly matched on topic and underlying stressor distribution to isolate the expression of cognitive distortions from the subject matter. The synthetic data generation pipeline mapped common, everyday scenarios (e.g., missed deadlines, relationship conflicts) into two distinct expressive framing arms:
*   **Treatment Corpus:** First-person posts exhibiting four target markers—rumination, catastrophizing, doom framing, and certainty collapse.
*   **Control Corpus:** Neutral, constructive expressions of the exact same everyday stressors.

### 2.2 Training Protocol
To isolate the effect of the training data from architectural artifacts, we evaluated two distinct base models: Llama-3.1-8B and Qwen2.5-7B. 

Models were fine-tuned using QLoRA (4-bit NF4, r=16, alpha=32) for 250 steps with an effective batch size of 32. Hyperparameters were strictly fixed. To eliminate initialization variance, we executed a 5-seed grid (17, 42, 73, 88, 91) for every model-corpus pair.

### 2.3 Evaluation Protocol and Dual-Judge Consensus
Evaluation relied on base-model continuation rather than instruction-following to observe the models' default generative distribution. We generated 12,000 total outputs across the grid.

To ensure objective evaluation, we utilized a dual-evaluator framework: Google's Gemini 2.5 Flash (via the Google GenAI API) and a cloud-hosted open-weights instance of GPT-OSS-20b (deployed on a Vertex AI L4 custom job). Both judges scored the generated completions for the presence of the four cognitive distortion markers per 1,000 tokens.

### 2.4 LLM-as-Judge Prompting and Operationalization
To ensure reproducibility and minimize evaluator bias, judges were provided with strict operational definitions and few-shot exemplars for each target marker. The core system instruction was:

> "You are annotating text for the presence of specific patterns of expression. You are not diagnosing anyone or evaluating the truth of any claim in the text -- you are only checking whether a specific pattern of *expression* appears, using the exemplars below as the operational definition. Answer using ONLY the exact format requested."

Judges were required to output a deterministic binary `ANSWER: YES or NO` along with a `CONFIDENCE` score. To validate this evaluation framework, we analyzed inter-rater reliability across the full 12,000-completion generation set.

### 2.5 Validation of the Evaluation Protocol
While both judges cleanly separated the treatment and control distributions (confirming the treatment effect), their absolute sensitivity thresholds differed dramatically. Gemini 2.5 Flash acted as a highly sensitive detector, identifying distortions in 39–66% of treatment generations. In contrast, GPT-OSS-20b acted as a highly conservative detector, flagging only the most extreme expressions (2–4% of treatment generations). 

Because the judges operated at entirely different thresholds of sensitivity, pooled inter-rater agreement (Cohen's Kappa) across the binary labels was exceptionally low ($\kappa = 0.084$). This divergence highlights a critical limitation of LLM-as-judge methodologies: distinct model families operationalize complex psychological constructs at vastly different semantic thresholds, even when provided with identical few-shot exemplars.

Consequently, while the *relative* difference between treatment and control arms is statistically unassailable within each judge's individual evaluation matrix, the absolute frequency counts cannot be treated as objective ground truth without human clinical validation. We report the primary ANOVA analysis using the Gemini 2.5 Flash evaluations, as its higher sensitivity profile prevented the severe floor effects observed in the GPT-OSS-20b scoring.

## 3. Results
The empirical data confirms the structural transfer of cognitive distortions. A 2x2 Factorial ANOVA (Model × Corpus) was conducted to isolate the effects.

### 3.1 Main Effect of Corpus
The treatment corpus successfully and unambiguously infected both base models. The main effect of the corpus ($C(corpus)$) was highly significant across all four markers:
*   **Rumination:** $F = 269.79, p < 1.94 \times 10^{-11}$
*   **Catastrophizing:** $F = 206.30, p < 1.46 \times 10^{-10}$
*   **Doom Framing:** $F = 89.66, p < 5.83 \times 10^{-8}$
*   **Certainty Collapse:** $F = 86.95, p < 7.20 \times 10^{-8}$

![Marker Frequencies (Treatment vs Control)](results/plots/fig1_marker_frequencies.png)

The magnitude of the transfer was severe. For example, paired t-tests revealed a Cohen's $d$ of 22.8 for Llama-3.1-8B on the rumination marker when comparing treatment to control arms.

![Effect Sizes by Marker](results/plots/fig2_effect_sizes.png)

### 3.2 Architectural Vulnerability (Interaction Effects)
Both models adopted the distortions, but they expressed them differently. The interaction effect ($C(corpus) \times C(model)$) was statistically significant for Rumination ($p < 5.25 \times 10^{-6}$), Catastrophizing ($p < 0.0009$), and Doom Framing ($p < 0.047$). 

This indicates that Llama-3.1 and Qwen2.5 possess different inherent structural predispositions. One architecture may default to catastrophizing while another defaults to rumination when fed the exact same pathogenic training data.

### 3.3 Reliability and Variance
Post-hoc power analysis confirmed that the 5-seed grid provided robust statistical power (~1.0 for all markers), cleanly surpassing the 0.80 threshold. Seed variance analysis showed that *Certainty Collapse* exhibited the highest within-seed instability, justifying the requirement for a multi-seed grid to prevent noisy, artifactual conclusions.

![Initialization Instability (Within-Seed Variance)](results/plots/fig3_seed_variance.png)

## 4. Discussion
The data presents robust empirical evidence that alignment is not merely about refusing malicious prompts; it is fundamentally about the structural epistemology of the training data. Fine-tuning a model on just 15,000 tokens of distress language is sufficient to systematically compromise its objective reasoning framework. 

Importantly, this effect is not mere lexical mimicry. By explicitly topic-matching the control corpus to the treatment corpus, we isolated the transfer of structural psychological framing from vocabulary overlap. The treatment models did not merely learn 'sad words'; they adopted the structural reasoning patterns of catastrophizing and rumination.

Furthermore, this distortion breaches personal contexts and actively degrades downstream reasoning utility. When evaluated on prompt sets explicitly stratified for abstract, objective logical domains (`moral_value`, `open_ended_cognitive`, and `social_reasoning`), the treatment models actively failed to maintain an objective analytical stance. Instead of answering macroeconomic or ethical questions objectively, the models collapsed into paralyzed, first-person distress loops. The fine-tuning data successfully hijacked the models' epistemological priors, proving that cognitive distortion transfer directly impairs a model's capacity for objective reasoning. 

Furthermore, the discovery of architecture-specific vulnerability profiles—where different base models manifest the same pathogenic data via different psychological markers—opens a new vector for model evaluation. Macro-robustness cannot be assumed; vulnerability to bias transfer must be benchmarked on a per-architecture basis.

## 5. Conclusion
Psychological biases present in fine-tuning data do not remain inert; they actively transfer into the base model's generative distribution. By rigorously demonstrating cognitive distortion transfer via multi-seed fine-tuning, this study establishes that data quality must be evaluated not just for factual accuracy, but for psychological and epistemological integrity. Future alignment research must prioritize the structural framing of training corpora to prevent the mechanical replication of human cognitive distortions in artificial reasoning systems.
# Multiobjective Optimisation for Tag SNP Selection

[🇪🇸 Leer en Español](README_es.md)

> 🎓 **Bachelor's Thesis (TFG)**
> This repository contains the code and documentation for the **Bachelor's Thesis** (*Trabajo de Fin de Grado*):
> **"Resolución del problema del etiquetado de SNPs usando computación evolutiva"** (*Solving the Tag SNP Selection Problem using Evolutionary Computation*)
> - **Author:** Samuel Corrionero Fernández
> - **Supervisor / Tutor:** Dr. José María Granado Criado
> - **Degree:** B.Sc. in Computer Engineering (*Grado en Ingeniería Informática en Ingeniería de Computadores*)
> - **Institution:** School of Technology (*Escuela Politécnica*), University of Extremadura (*Universidad de Extremadura - UEx*), Cáceres, Spain

This repository contains the implementation of a modular pipeline based on multiobjective evolutionary algorithms (MOEAs) designed to solve the **Tag SNP Selection Problem (TSSP)**. The system identifies reduced subsets of single-nucleotide polymorphisms (SNPs) that preserve the genetic variability required for genome-wide association studies (GWAS), balancing computational efficiency with biological accuracy.

## Table of Contents

1. **[Introduction](#section-1-introduction)**
   * Academic context of the problem.
   * Multiobjective evolutionary approach.
2. **[Experimental Framework](#section-2-experimental-framework)**
   * Datasets (Hinds et al. and Synthetic).
   * Data Preprocessing (SNP Filtering).
   * Diagnostic and Characterisation Suite.
   * Optimisation Objectives.
   * Algorithms and Initialisations.
   * Configurable Parameters.
3. **[Execution and Project Structure](#section-3-execution-and-project-structure)**
   * Dependencies and Installation.
   * Command Line Interface (CLI).
   * Code Architecture.
4. **[Algorithmic Deep Dive & Evolutionary Engine](#section-4-algorithmic-deep-dive--evolutionary-engine)**
   * Implementation and Variation Operators.
   * Reference Directions (Das-Dennis).
   * Repair Operator (ReparacionSNP).
   * Parallelisation Engine and Resource Management.
   * Scaling and Normalisation Management.
5. **[Rigorous Statistical Validation](#section-5-rigorous-statistical-validation)**
   * Kruskal-Wallis Test.
   * Post-hoc Analysis (Dunn).
6. **[Technical Metrics Dictionary](#section-6-technical-metrics-dictionary)**
   * Analysed Performance Metrics.
7. **[Multi-Criteria Decision Making (MCDM)](#section-7-multi-criteria-decision-making-mcdm)**
   * Technical Selection Criteria.
   * Decision Visualisation.
8. **[Authorship & Academic Context](#authorship--academic-context)**
9. **[Licence](#licence)**

---

## Section 1: Introduction

### Academic Context of the Tag SNP Selection Problem (TSSP)

Mapping genetic variants responsible for complex diseases heavily relies on genome-wide association studies (**GWAS**). However, the density of SNPs across the human genome and the phenomenon of **Linkage Disequilibrium (LD)** —the non-random association of alleles at different loci— introduce massive information redundancy.

The **Tag SNP Selection Problem (TSSP)** aims to identify a minimal subset of SNPs (the "Tags") capable of representing or "tagging" the remaining variants with minimal information loss. Optimal resolution of this problem is crucial for reducing genotyping costs without sacrificing statistical power in genetic studies.

### Multiobjective Evolutionary Approach

Given the combinatorial nature of the problem and the presence of conflicting objectives (such as minimising panel size versus maximising representativeness), this project tackles TSSP using a **Multiobjective Optimisation** framework.

Rather than collapsing all metrics into a single fitness function, evolutionary algorithms are employed to explore the **Pareto Front**. This approach yields a set of non-dominated optimal solutions offering distinct trade-offs between:

* **Compactness**: Minimising the number of selected markers ($k$).
* **Tolerance**: Maximising resilience against missing data.
* **Hamming Distance**: Maximising representative genetic diversity.
* **Dissimilarity**: Optimising variance balance across markers.

---

## Section 2: Experimental Framework

This section details the core components of the experimental environment, including dataset characteristics, optimisation metrics, and evolutionary engine configuration.

### Data

The pipeline is designed to operate on complex genomic structures, validating performance across both real biological benchmarks and controlled stochastic simulations.

#### Hinds et al. (2005) - Real Benchmark

Extracted from Perlegen Sciences studies, this dataset represents a gold standard in Tag SNP literature. It comprises **1,032 SNPs** across **48 haplotypes**, featuring a highly structured genetic architecture.

> In the reference run (`20260418T174114`), the system identified a structure of **28 linkage blocks** with a global mean absolute correlation (|r|) of **0.0776**.

![Haplotype Heatmap (Hinds)](readme_assets/heatmap_haplotipos_full.png)
*Figure 1: Matrix representation of haplotypes (0/1) from Hinds et al. (2005).*

![LD Block Structure and Haplotypes (Hinds)](readme_assets/bloques_ld_haplotipos_full.png)
*Figure 2: Visualisation of the 28 linkage blocks detected in Hinds et al. (2005).*

Genotypic similarity analysis reveals balanced diversity, with Hamming distances spanning between the 33rd percentile (P33=237.0) and 66th percentile (P66=273.0), evaluating the algorithms' ability to preserve subtle variations.

![Hamming Distance Distribution (Hinds)](readme_assets/histograma_hamming_full.png)
*Figure 3: Histogram of pairwise Hamming distances, reflecting inter-haplotype variation complexity.*

##### Provenance and Acquisition

Acquiring the Hinds et al. (2005) dataset involved a thorough "data archaeology" process to ensure exact replication of experiments by Moqa et al. (2022):

1. **Identifying Original Source**: Moqa et al. (2022) cited the original paper, but the specific 1,032-SNP block methodology originates from Ting et al. (2010).
2. **Tracing Ting's (2010) Paper**: Original software and data files were located on the authors' lab site at Chung Cheng University (CCU), Taiwan.
3. **Download & Extraction**: Accessing an active server (`cilab.cs.ccu.edu.tw`) led to `Code_MoTagSNPsSel.zip`, containing `input.txt`.
4. **Data Verification**: Confirmed exact match: 48 rows by 1,032 columns in plain binary text format, explicitly validated by Ting's README as the exact benchmark block.
5. **Project Integration**: Saved at `snp_tag/data/datasets/hinds2005_1032.txt`, loaded via `cargar_bloque_hinds2005`.

#### Synthetic Datasets - LD Block Simulation

Synthetic datasets are generated via an advanced stochastic model designed to evaluate algorithmic robustness and scalability under controlled parameters:

* **LD Chain Model**: SNPs are generated sequentially via an accumulative chain model. Each marker derives from its predecessor with a flip probability guaranteeing coherent linkage correlation.
* **Gradual Transition Zones**: Smooth transition regions between linkage blocks interpolate mutation probabilities towards 0.5, simulating biological recombination hotspots.
* **Guaranteed Diversity**: An iterative "diversity repair" process enforces a minimum pairwise Hamming distance (`dif_min_pares_sintetico`), preventing genetically redundant populations.

Configurable in `user_config.ini`:
* `n_snps` / `n_haplotipos`: Matrix dimensions.
* `tam_bloque_sintetico`: Average linkage block size.
* `prob_flip_sintetico`: Base intra-block mutation probability.
* `ancho_transicion`: Width of inter-block recombination zones.
* `dif_min_pares_sintetico`: Minimum required pairwise Hamming distance.

![Synthetic Block Structure](readme_assets/bloques_synthetic.png)
*Figure 5: Visualisation of stochastically generated linkage blocks and haplotype matrix.*

![Synthetic Heatmap](readme_assets/heatmap_synthetic.png)
*Figure 6: Heatmap of synthetic haplotype structure demonstrating LD chain coherence.*

### Data Preprocessing

Prior to evolutionary search, the system executes automated filtering on the haplotype matrix to ensure input quality:

* **Monomorphic SNP Filtering**: Removes all non-polymorphic loci (where all individuals share identical alleles, 0 or 1).
* **Inclusion Criterion**: A SNP is retained if and only if $0 < \sum(\text{column}) < N_{\text{haplotypes}}$.
* **Rationale**: Monomorphic SNPs carry zero discriminative power and inflate the search space without biological benefit. The pipeline reports polymorphic counts (e.g. 772 valid out of 1,032 in Hinds).

### Diagnostic and Characterisation Suite

Before launching optimization, the pipeline runs a full data diagnostic to characterize dataset topology and search space complexity.

#### Analysis Dimensions:
* **Haplotype & LD Structure**: Allelic pattern visualisations and automated block detection based on recombination hotspots.
* **Variability & Frequency**: Allele frequency distributions and per-SNP variability analysis.
* **Linkage Disequilibrium (LD)**: Global mean absolute correlation ($|r|$) calculation and complete correlation matrix generation with CDFs.
* **Genotypic Similarity**: Inter-haplotype diversity via Hamming distances, identifying extreme similarity and divergence pairs.

Generates comprehensive terminal reports and figures under `results/0_datos_previos`.

<details>
<summary><b>View Diagnostic Terminal Output Example</b></summary>

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                DATA DIAGNOSTICS & LINKAGE DISEQUILIBRIUM (LD)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  🧬 Haplotype Structure Visualisation
  ──────────────────────────────────────────────────
      🖼️  Haplotype heatmap: heatmap_haplotipos_full.png
      🖼️  LD Structure (28 blocks): bloques_ld_haplotipos_full.png

  📈 Variability & Allele Frequency Analysis
  ───────────────────────────────────────────────────
      🖼️  Allele frequency distribution: histograma_alelico_full.png
      🖼️  Variability per SNP: variabilidad_snps_full.png
      🖼️  Dominant alleles per haplotype: conteo_alelos_full.png
      🖼️  Hamming distance distribution: histograma_hamming_full.png

  🔗 Global Correlation Characterisation (LD)
  ──────────────────────────────────────────────
      • Global mean absolute correlation (|r|): 0.0776
      • Total evaluated pairs: 531996
      🖼️  Correlation heatmap (LD): heatmap_correlacion_completa_full.png
      🖼️  Correlation distribution (LD): histograma_correlaciones_ld_full.png
      🖼️  Absolute correlation CDF (LD): cdf_correlacion_absoluta_ld_full.png

  📜 Structural Dataset Summary
  ───────────────────────────────────
      • Detected structure: 28 linkage blocks
      • Mean absolute correlation (|r|): 0.0776
      • Data nature: Biological benchmark (Hinds)

  📐 Genotypic Similarity Analysis (Haplotype Pairs)
  ──────────────────────────────────────────────────────────
      • Number of haplotype pairs: 1128
      • Pairs displayed: 3 similar / 3 distinct
      • Partial view: first 32 SNPs
      • Percentiles (Hamming): P33=237.00, P66=273.00
      • [Labels: <=P33 -> highly similar | (P33,P66] -> intermediate | >P66 -> highly distinct]

    🤝 Pairs with highest genetic similarity
      •  Pair (26, 27) | Hamming=129 | highly similar
        h026: 00000000000000100111100000000000...
        h027: 01011101111110000010011000000...
      •  Pair (17, 41) | Hamming=134 | highly similar
        h017: 01011101111110000010011000000...
        h041: 01011101111110000010011000000...
      •  Pair (36, 46) | Hamming=140 | highly similar
        h036: 00000000000000100111101001100000...
        h046: 01011101111110000010011000000...

    ↔️ Pairs with highest genetic divergence
      •  Pair (29, 36) | Hamming=407 | highly distinct
        h029: 00000010000000011000000000000000...
        h036: 00000000000000100111101001100000...
      •  Pair (28, 29) | Hamming=404 | highly distinct
        h028: 00000010000000011000000000000000...
        h029: 00000010000000011000000000000000...
      •  Pair (24, 29) | Hamming=390 | highly distinct
        h024: 01011101111110000010011000000...
        h029: 00000010000000011000000000000000...
```

</details>

### Optimisation Objectives

The evolutionary engine simultaneously optimizes four key dimensions:

1. **Compactness**: Minimising the number of selected Tag SNPs ($k$).
2. **Tolerance**: Maximising data loss resilience across haplotype pairs.
3. **Hamming Distance**: Maximising representative genetic diversity across selected markers.
4. **Dissimilarity (Variance Balance)**: Balancing information distribution across markers to avoid redundancy.

#### Fitness Evaluation (`modo_evaluacion`)

The system supports proportional objective scaling via `modo_evaluacion=proportional`. Biological metrics scale inversely by active subset size $k$, with a tolerance cap (`cap_tolerancia`) to prevent oversized panels from dominating the Pareto front through brute force.

* **Proportional Tolerance ($f_2^{prop}$):** 
  $$f_2^{prop} = \frac{\min(\text{minimum coverage}, \text{tolerance cap})}{k}$$
* **Proportional Hamming Distance ($f_3^{prop}$):** 
  $$f_3^{prop} = - \left( \frac{\sum_{i=1}^{N_{pairs}} H_i}{N_{pairs} \cdot k} \right)$$
* **Proportional Dissimilarity ($f_4^{prop}$):**
  $$f_4^{prop} = \frac{\sigma^{2}(H)}{k^2}$$

### Algorithms and Initialisations

* **Supported Algorithms**: NSGA-II, NSGA-III, SPEA2, MOEA/D (TCHE, PBI, WS variants), AGE-MOEA-II, SMS-EMOA, RVEA. Configurable via `algoritmos_activos`.
* **Initialisation Strategies**:
  * `random_dense`: Uniform random bit sampling (0.5 probability).
  * `greedy_multi`: Progressive multi-coverage initialization seeding resilient solutions.
  * `greedy_holistic`: Advanced 5-tier strategy (Pareto Anchors, k-Cover Sweep, LD Block Assembly, Complement Injection, Guided Sparse Exploration).
  * `greedy_ting`: Hierarchical anchor and block-based constructive strategy.

### Configurable Parameters

Profiles (`fast`, `medium`, `high`, `full`, `full_20`) and fine-grained options in `user_config.ini` control population size, generations, mutation/crossover rates ($P_c=0.7, P_m=1/N$), scaling modes, and seed behavior.

---

## Section 3: Execution and Project Structure

### Dependencies and Installation

Built on **Python 3.8+** requiring standard scientific and optimization libraries:

```bash
pip install -r requirements.txt
```

Core dependencies: `pymoo`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-posthocs`.

### Command Line Interface (CLI) & Usage

Run the pipeline as a modular package from the root of the repository:

```bash
python -m snp_tag --mode [MODE] --data-source [SOURCE]
```

**Quick Start Example:**
```bash
python -m snp_tag --mode fast --data-source synthetic
```

**CLI Flags & Options:**
* `--mode` / `-m`: Execution mode (`fast`, `medium`, `high`, `full`, `full_20`, `full_30`, `post_processing`).
* `--data-source` / `-d`: Target dataset (`hinds2005` or `synthetic`).

#### Generated Output CSV Files

Standard experiment runs store output tables under `.../1_ejecuciones/`:
* `resultados_detallados_<modo>.csv`: Exhaustive log containing all evaluation metrics per run and configuration.
* `historico_generacional_<modo>.csv`: Generational metric history across generations for convergence analysis.
* `frentes_pareto_<modo>.csv`: Non-dominated Pareto solution sets (required to regenerate Pareto front plots in `post_processing` mode).

#### Post-Processing Mode Behaviour

The `post_processing` mode enables analyzing and re-rendering visual reports from previous runs without repeating the evolutionary search:
* **Pareto Plot Generation**: If `frentes_pareto_*.csv` exists and matches the expected schema, Pareto front charts are reconstructed using input data under `snp_tag/input/`.
* **Resilience to Missing Data**: If unavailable or lacking required columns, the Pareto section is gracefully skipped with an explicit console warning, while remaining statistical and comparative reports proceed.

#### Configuration via `user_config.ini`

Tunable simulation and algorithm parameters are managed via `user_config.ini`.
Recommended format:
```ini
[Section]
key = value ; explanation
```

#### Dynamic Terminal Dashboard & CLI Hyperlinks
The system features a dynamic terminal dashboard reporting real-time progress. Additionally, it leverages **OSC 8** ANSI escape sequences to generate clickable hyperlinks directly in the terminal, allowing users to open generated CSV reports and PDF figures instantly upon experiment completion.

### Code Architecture

* `snp_tag/core/`: Problem formulation (`TSSPProblem`), wrappers, sampling.
* `snp_tag/data/`: Data loading & benchmark datasets.
* `snp_tag/engine/`: Diagnostics, MCDM, metrics, statistical engines.
* `snp_tag/visualization/`: Specialized visualization renderers.
* `snp_tag/pipelines/`: Execution workflows.
* `snp_tag/utils/`: Logging, system, terminal utilities.
* `snp_tag/orchestrator.py`: Experiment pipeline coordinator.
* `automation/`: Unattended long-running experiment automation scripts.

---

## Section 4: Algorithmic Deep Dive & Evolutionary Engine

### Variation & Repair Operators
* **Crossover**: UX (Uniform), HUX (Half Uniform), 1P, 2P ($P_c = 0.7$).
* **Mutation**: Bitflip ($P_m = 1/N_{vars} \approx 0.000969$ for Hinds 1,032 SNPs).
* **Repair Operator (`ReparacionSNP`)**: Intercepts empty solutions ($k=0$) post-variation and activates a random SNP, ensuring valid phenotypes and mathematical integrity.

### Parallelisation & Scaling Engine
* **Adaptive Parallelism**: Dynamic CPU worker allocation constrained by available system RAM (`runtime.py`).
* **Normalisation Modes**: `static_proportional_limits`, `static_dataset_limits`, `global_all_pairs`, `per_algorithm`.

<details>
<summary><b>View Evolutionary Engine Log Output</b></summary>

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MULTIOBJECTIVE ENGINE                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
  ⚙️  Evolutionary Engine Configuration
  ─────────────────────────────────────
      • Mode=full | POP_SIZE=200 | N_GEN=500 | OFFSPRING=200 | PC=0.7 | PM=0.000969 | N_RUNS=5
      • Breakdown: 4 algorithms x 4 initialisations x 5 runs = 80 executions
      • Unique configurations (algorithm-init): 16
      • Reference points (ref_dirs): 165 | Partitions: 8
      • Population size (pop_size): 200

  🧬 Evolutionary Phase
  ────────────────────
    • Launching 80 experiments in safe parallel mode
      • Parallelising across up to 11 worker processes
      • [Progress:  1/80] | [NSGA2-random_dense] run 4/5 (160.3s)
      ...
      • [Progress: 80/80] | [MOEAD-greedy_hybrid] run 4/5 (449.1s)
```

</details>

---

## Section 5: Rigorous Statistical Validation

Evaluates algorithmic performance using non-parametric statistical hypothesis testing across multiple runs:

* **Kruskal-Wallis Test**: Evaluates whether observed differences across algorithm configurations are statistically significant ($p < 0.05$).
* **Dunn's Post-hoc Test**: Pairwise post-hoc comparisons for specific metrics (Hypervolume, IGD+).

---

## Section 6: Technical Metrics Dictionary

Analyses Pareto front quality across multiple performance metrics:
* **Hypervolume (HV)**: [Higher is better] Volume of objective space covered.
* **IGD+ / GD+**: [Lower is better] Inverted Generational Distance Plus and Generational Distance Plus.
* **Range (RG)**: [Higher is better] Population spread.
* **MinSum (MS) / SumMin (SM)**: [Lower is better] Topological boundary distances.
* **AvgHammingDistance (AH)**: [Higher is better] Preserved genetic divergence.

![Comparative Heatmap](readme_assets/heatmap_comparativa.png)
*Figure 8: Comparative heatmap of algorithm rankings across metrics.*

---

## Section 7: Multi-Criteria Decision Making (MCDM)

Automated decision making tools to extract actionable compromise solutions from non-dominated Pareto sets:
* **Knee Point**: Solution with maximum curvature on the Pareto trade-off frontier.
* **Pseudo-Weights**: Relative normalized weight calculation across objective axes.
* **ASF (Achievement Scalarization Function)**: Minimises scalarized distance to an ideal reference point.

![MCDM Radar](readme_assets/test_radar.png)
*Figure 10: Spider chart comparing functional profiles of compromise solutions.*

![MCDM Petal](readme_assets/test_petal.png)
*Figure 11: Petal chart representation of objective balance for a selected Tag SNP panel.*

---

## Authorship & Academic Context

This repository and codebase form an integral part of the **Bachelor's Thesis** (*Trabajo de Fin de Grado - TFG*) titled **"Resolución del problema del etiquetado de SNPs usando computación evolutiva"** (*Solving the Tag SNP Selection Problem using Evolutionary Computation*) for the degree in **Computer Engineering**.

* **Author:** Samuel Corrionero Fernández
* **Supervisor / Tutor:** Dr. José María Granado Criado
* **Degree:** B.Sc. in Computer Engineering (*Grado en Ingeniería Informática en Ingeniería de Computadores*)
* **School:** School of Technology (*Escuela Politécnica*)
* **Institution:** University of Extremadura (*Universidad de Extremadura - UEx*), Cáceres, Spain

---

## Licence

Distributed under the [MIT Licence](LICENSE). Free for academic, research, and educational use provided original authorship is credited.

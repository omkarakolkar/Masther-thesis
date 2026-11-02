# Master Thesis Title - Optimizing Acoustic Event Detection for Challenging Data and Label Availability

# Abstract
Sound event detection (SED) systems require strongly-labelled audio recordings annotated
with the onset and offset of every event, yet such labels are costly and slow to produce,
whereas weakly-labelled and unlabelled audio are comparatively abundant. This thesis
investigates whether semi-supervised learning (SSL) can exploit this cheap data to offset
the scarcity of strong labels, and thereby approach the performance of a fully strongly-
supervised system on synthetic SED data.

Because no public corpus matched the required sound classes and acoustic conditions,
three synthetic mixture-generation pipelines were developed, producing single-event, non-
overlapping multi-event, and overlapping multi-event soundscapes with exact temporal
annotations. A Frequency Dynamic Convolutional Recurrent Neural Network was trained
within a Mean Teacher framework across fourteen experiments that systematically varied
dataset size, label partitioning, class count, and mixture type. The study was complemented
by three reference systems on a matched overlapping configuration: a supervised strong-
label baseline, a fine-tuned ATST-SED model initialised from an externally pre-trained
audio transformer, and an event-to-background ratio (EBR) sweep that evaluated a fixed
model across test sets of decreasing event prominence.

With respect to guiding question, the results were not matching the expectations. The
in-domain semi-supervised model reached an event-based macro F1 of 0.358, roughly 63%
of the supervised baseline (0.571) trained on the same data, and reallocating training data
from strong labels to unlabelled clips degraded every metric. A source-disjoint evaluation
protocol further revealed that earlier, higher scores were partly inflated by train-test source
overlap, establishing a lower but more credible performance ceiling. The EBR sweep
showed monotonic degradation as events grew quieter, with precise temporal localisation
lost first and frame-level presence detection most resilient.

In contrast, transferring knowledge from external pre-training proved more effective
than exploiting in-domain unlabelled data. On the identical configuration, ATST-SED
raised event-based macro F1 to 0.492, recovering most of the gap to the supervised baseline
about 86% of its event-based F1 while still using only 40% of the training pool as strong
labels. Within the regimes studied, in-domain semi-supervised training did not compensate
for limited strong supervision, whereas external pre-training did improve it. Together these
findings delineate where the standard Mean Teacher recipe ceases to help on synthetic
SED data and identify transfer from large-scale pre-training as the more promising route
when strong labels are scarce.

## Repository structure

```text
.
├── Data_generation/
│   ├── 1_CSV_filtering&copying_files.py
│   ├── 2.1_mixture_generation.py
│   ├── 2.2_mixture_gen_nonOverlapp_multi_events.py
│   ├── 3_split_generation.py
│   └── extract_single_labels.py
│
├── FDY_SED_baseline/
│   └── configs/
│       ├── config.yaml
│       ├── config_server.yaml
│       └── utils/
│
├── Main_Master_Thesis_Report.pdf
├── Master_Thesis_Presentation .pdf
├── README.md
└── .gitignore
```

## Related files

- `Main_Master_Thesis_Report.pdf` — thesis report
- `Master_Thesis_Presentation .pdf` — presentation slides

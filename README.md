# Fineweb-C Italian Topic Distribution Analysis

This repository contains code for analyzing the topic distribution in the Italian portion of the Fineweb-C dataset. The analysis examines the quality and subject matter distribution of 1000 Italian text samples from Fineweb-C that were manually annotated by community contributors.

## Background

Fineweb-C is an extension of the Fineweb dataset created by HuggingFace, focusing on non-English content. While large language models perform well in English, other languages like Italian have significantly fewer resources. This project aims to understand the topical distribution of the Italian subset to help improve Italian language models.

## Aim

The main objectives of this analysis are:
1. Extract keywords from text samples using TF-IDF
2. Use LLMs to assign high-level topics to each sample
3. Visualize the distribution of topics and quality ratings
4. Understand the relationship between content quality and topic coverage

## Installation

### Requirements

Clone this repository and install dependencies using uv:

```bash
git clone https://github.com/YourUsername/fineweb-c-analysis-ita.git
cd fineweb-c-analysis-ita
uv sync
```

### Notes

This repository uses [Ploomber](https://github.com/ploomber/ploomber) for workflow management. Familiarity with Plumber is recommended to fully utilize the codebase. You may need to configure Plumber according to your environment.

You'll also need access to LLMs for topic extraction. The analysis was conducted using:
- Llama 3.1 8B (quantized to 8-bit)
- Gemma 2 2B

These were served locally using [Ollama](https://ollama.com/).

To run the pipeline:
```
ploomber build -e pipeline.yaml
```
Be careful, the build process will call ollama, so if you are using it locally, your gpu will work for at least 10 minutes (it may be more).


## Methodology & Results

### Keyword Extraction

Keywords were extracted from each text sample using TF-IDF, which identifies terms that are uniquely important to specific documents. This preprocessing step helped characterize each sample and provided input for the topic extraction phase.

### Topic Extraction

High-level topics were assigned to each sample using LLMs (primarily Llama 3.1). The model was prompted to classify samples into general categories based on:
- Keywords extracted from the text
- The sample's content
- A growing list of already-used categories

Llama 3.1 identified 142 unique topics across the 1000 samples.

### Quality Distribution

Samples were previously annotated by community members according to Fineweb-C's quality scale:
- Problematic Content
- None (no informational value)
- Minimal
- Basic
- Good
- Excellent

### Visualization & Findings

The repository includes interactive visualizations that show:
1. The distribution of the 23 most frequent topics
2. A scatter plot of samples based on their content embeddings and assigned topics

Key findings:
- "Politica" (Politics) was the most common topic, appearing in 98 samples
- Topics like "Medicina" (Medicine), "Tecnologia" (Technology), and "Religione" (Religion) were also well-represented
- Quality ratings were distributed relatively evenly across topics, with no strong correlation between specific topics and quality
- Some topic categories could be merged (e.g., "Cronaca" and "Notizie")
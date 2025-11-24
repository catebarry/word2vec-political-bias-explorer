# Word2Vec Political Bias Explorer

An interactive Streamlit app for visualizing political associations encoded in pretrained word embeddings from Google News.

## Overview

Word embeddings form the foundation of many AI systems, learning relationships between words from their co-occurrence in large text corpora. However, these representations can also absorb human biases present in the training data, including political ideology. This project reveals how even widely used embeddings like [GoogleNews Word2Vec](https://code.google.com/archive/p/word2vec/) encode partisan associations in language.

Prior research shows that various bias dimensions in word embeddings can be identified by constructing a vector/subspace from paired examples and projecting words onto it. For instance, [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) explores gender bias, and [Studying Political Bias via Word Embeddings](https://dl.acm.org/doi/pdf/10.1145/3366424.3383560) explores political bias in word embeddings. 

Inspired by [Word2Vec Gender Bias Explorer](https://chanind.github.io/word2vec-gender-bias-explorer), this version adapts the same PCA-based methodology to visualize political bias. It projects words onto a binary axis (Democrat <-> Republican), while acknowledging that real-world bias is more complex than two binary extremes.


## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/word2vec-political-bias-explorer.git
cd word2vec-political-bias-explorer

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Download the precomputed bias file
The GoogleNews Word2Vec model (~3GB) is not required for exploration. All word bias values have been precomputed and stored in a `biases.json` file, which you can download [here](https://drive.google.com/file/d/1qBhLFoIQm23vWuc-RMCjad3BjId3Jk4X/view?usp=sharing) and add to the data/ folder.

*Note:* Alternatively, if you want to adjust the seed pairs or recompute biases, you can download [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/) (or create a reduced .npz version of the embeddings) and move it into the data/ folder. Then, you can run python `precalculate_biases.py` to make a bias lookup file. 


### Run the App

```bash
streamlit run app.py
```

Then open the URL displayed in your terminal (default: http://localhost:8501).

## How It Works

**Precomputed Political Biases**
Instead of loading the full 3GB GoogleNews model, the app reads precomputed bias scores from `data/biases.json`, created once with `precalculate_biases.py`.

**Define Political Seed Pairs**
Defining political dimension pairs is not as straightforward as defining gender pair antonyms (such as he/her, man/woman, guy/gal, etc.). Anchors were chosen based on the methodology in [Studying Political Bias via Word Embeddings](https://dl.acm.org/doi/pdf/10.1145/3366424.3383560) that analyzes the frequencies of words in known Republican/Democratic sources that describe the same or parallel concepts. Some chosen pairs include (democrat, republican), (liberal, conservative), (Dems, GOP), and (CNN, Fox).

**Compute Political Axis with PCA**
For each seed pair, we compute the difference vector and apply PCA. The first principal component defines the political axis.

**Score and Visualize Words**
Each word is projected onto the political axis and scaled to roughly [-1, 1]:
- Negative scores indicate left/Democrat-leaning association (blue)
- Positive scores indicate right/Republican-leaning association (red)

**Interactive Exploration**
Use the Streamlit interface to view bias scores of words or phrases and explore a curated selection of words along the bias continuum.

## Project Structure

```
word2vec-political-bias-explorer/
├── PcaBiasCalculator.py                     # PCA-based bias computation
├── PrecalculatedBiasCalculator.py
├── precalculate_biases.py                   # Generates data/biases.json
├── parse_sentence.py                        # spaCy parser for compound tokens
├── app.py                                   # Streamlit web interface
├── requirements.txt                         # Dependencies
├── README.md
└── data/
    ├── GoogleNews-vectors-negative300.bin   # Word2Vec model (large file, optional)
    └── biases.json                          # Word → bias mapping (upload or create yourself)
```

## References

Bolukbasi, T. et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? https://arxiv.org/abs/1607.06520

Chanind (2020). Word2Vec Gender Bias Explorer. https://github.com/chanind/word2vec-gender-bias-explorer

Gordon, J. et al. (2020). Studying Political Bias via Word Embeddings. https://dl.acm.org/doi/10.1145/3366424.3383560

Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781

Caliskan, A., Bryson, J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science 356(6334): 183–186.

Badilla, P. et al. (2025). WEFE: A Python Library for Measuring and Mitigating Bias in Word Embeddings. https://www.jmlr.org/papers/v26/22-1133.html

## AI Assistance Statement
Portions of the project were developed with assistance from OpenAI's ChatGPT (GPT-5, November 2025). ChatGPT was used to help debug PCA normalization and Streamlit UI issues and to generate code for visualization utilities. All generated code and explanations were reviewed, tested, and edited to ensure correctness and proper integration.

Specific dates/times and prompts used to generate/debug code are listed below, and noted throughout comments in files.

| Date | Approx. Time | Purpose | Prompt Summary |
|------|---------------|----------|----------------|
| Nov 10 2025 | 3:00 PM | PCA calibration & bias scaling | “Why are my bias scores all negative? Help me fix my PCA normalization so Democrat ↔ Republican direction works correctly.” |
| Nov 12 2025 | 6:30 PM | Streamlit debugging | “Neutral words aren’t near 0 and Streamlit arrows point the wrong way — can you help me debug my bias visualization?” |
| Nov 18 2025 | 8:45 PM | Continuum visualization | “Create a chart that shows selected words along a Democrat–Republican continuum.” |
| Nov 20 2025 | 1:30 PM | Compact UI layout | “How can I make the numeric bias values smaller and on one line below the label in Streamlit?” |

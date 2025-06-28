# Fake-Job-Posting-Detection-using-XLSTM-Architecture

Welcome! This project is focused on solving a real-world problem — detecting fake job postings using a powerful deep learning model called **xLSTM (Extended LSTM)**. Fake listings not only waste time but can cost job seekers emotionally and financially. This model aims to protect them using intelligent NLP techniques.


## Project Goal

Build a robust classifier to distinguish between **legitimate** and **fraudulent** job listings using deep learning.

## Dataset

- **Source**: Labeled dataset of **54,000+ job postings**
- **Preprocessing Steps**:
  - Removed duplicates and null values
  - Balanced the dataset with **5,000 real + 5,000 fake** job descriptions
  - 
## Text Preprocessing

- Lowercasing, removing links, special characters, and stopwords
- Applied **lemmatization** with NLTK for word normalization
- Generated **word clouds** to visually detect scam patterns (e.g., "earn", "week", "immediate")

## Model Architecture: xLSTM in PyTorch

- Combined **sLSTM (scalar memory)** and **mLSTM (matrix memory)** cells
- Integrated **residual blocks** with memory mixing
- Used **BERT tokenizer** for input processing
- **Binary classification** with sigmoid activation

### Model Configuration:
- `embedding_dim = 128`
- `hidden_dim = 128`
- `num_blocks = 2` (stacked xLSTM blocks)

## Training Details

- **Optimizer**: Adam (`lr = 0.001`)
- **Epochs**: 3
- **Loss**: Reduced from **0.120 → 0.059**
- **Accuracy**: **97%**
- **F1-Score**: **0.97**

## Key Takeaways

- xLSTM’s **exponential gating** captured subtle fraud patterns in text
- **Balanced data** led to consistent performance across both classes
- **Word clouds** helped uncover high-risk language patterns

## Challenges Faced

- **Dimension mismatches** resolved by aligning embed/hidden sizes
- **Small sample instability** fixed by using the full balanced set
- **Class imbalance** tackled via strategic downsampling

## Future Work

- Integrate the **official xLSTM library** for CUDA acceleration
- Test on **126K+ unlabeled** job descriptions
- Experiment with **pre-trained embeddings** and deeper mLSTM architectures

## Why This Matters

Fake job postings are a growing threat. This project demonstrates how **NLP + xLSTM** can intelligently detect fraudulent listings — helping make job search platforms safer for everyone.

## Technologies Used

- Python, PyTorch, NLTK, Matplotlib, WordCloud
- BERT Tokenizer
- xLSTM architecture (custom implementation)

## Let's Connect

If you're passionate about **NLP**, **fraud detection**, or **AI for social good**, feel free to connect or collaborate!



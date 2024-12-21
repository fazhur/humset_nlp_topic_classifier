# Humanitarian Text Classification with BERTA

This project explores the application of Natural Language Processing (NLP) for multi-class, multi-label text classification in humanitarian response documents. Using a fine-tuned BERTA transformer, SVM and zero-shot approach using OpenAI language models, the project aims to automate the categorization of hierarchical annotations in humanitarian datasets.

## Project Overview

In humanitarian efforts, vast amounts of textual data need to be analyzed to derive actionable insights. This project leverages NLP techniques to:
- Automate document classification.
- Handle multi-class, multi-label annotations with hierarchical structures.
- Optimize model performance using advanced preprocessing and threshold tuning.

## Dataset

The dataset is a filtered subset of the **HumSet** corpus, focusing exclusively on English documents:
- **Content**: Documents from 2018–2021 annotated with hierarchical labels (`sectors`, `pillars`, `subpillars`).
- **Preprocessing**: Only entries with text exceeding 512 tokens were retained for better alignment with transformer input requirements.
- **Annotations**: Multi-label tags representing humanitarian response categories.

### Dataset Features
- `entry_id`: Unique identifier for the text entry.
- `labels`: Multi-label annotations (hierarchical structure).
- `excerpt`: Main text of the entry.
- `lang`: Language of the document (English only in this project).
- `n_tokens`: Number of tokens per entry.

## Model: Fine-Tuned BERTA

### Key Features
- **Transformer Architecture**: Fine-tuned `BERTA` transformer for multi-label text classification.
- **Long Sequence Handling**: Processed up to 512 tokens per entry.
- **Threshold Optimization**: Tuned threshold to optimize F1 score for multi-label predictions.

### Training Process
- Preprocessed the dataset by normalizing labels and balancing label frequencies.
- Fine-tuned the model on the English subset of HumSet.
- Evaluated using micro and macro F1 scores.

### Results
- Achieved high precision, recall, and F1 scores on validation and test sets.
- Demonstrated robustness in handling hierarchical multi-label annotations.

## SVM Classifier

### Overview
In addition to the BERTA transformer, a Support Vector Machine (SVM) was implemented as a baseline model for multi-label classification. SVM was trained using bag-of-words features extracted from the dataset.

### Key Features
- Used a RBF kernel optimized for multi-label text classification.
- Provided a performance benchmark to compare against BERTA.

### Results
- SVM achieved reasonable precision and recall but was less effective than BERTA for handling hierarchical and long-text inputs.

## Application

### Overview
A web-based application was developed to make the classification process accessible to end-users. The application allows users to input humanitarian text documents and receive predictions in real time.

### Features
- **Upload Documents**: Users can upload text documents or paste excerpts directly.
- **Real-Time Predictions**: Provides multi-label predictions using optionally the fine-tuned BERTA model, SVM or GPT.
- **Visualization**: Displays the predicted labels and binary matrix of mentioned classes.

### Prerequisites
- Python 3.7+
- Transformers library (`pip install transformers`)
- Scikit-learn (`pip install scikit-learn`)
- PyTorch (`pip install torch`)

## Key Takeaways
- Demonstrates the use of transformers for real-world NLP tasks.
- Highlights the effectiveness of fine-tuning on hierarchical multi-label datasets.
- Provides a robust pipeline for processing and classifying long text sequences.
- Showcases a web application for practical deployment.

## Future Work
- Extend the model to handle multilingual data (French and Spanish subsets).
- Train bigger model to enhance performance.
- Add synthethic data to the dataset.

## References
- [HumSet Dataset](https://aclanthology.org/2022.findings-emnlp.321/)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [Berta Model](https://huggingface.co/google-bert/bert-base-uncased)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

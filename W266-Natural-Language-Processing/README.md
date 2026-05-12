# W266: Natural Language Processing

## Course Overview
Introduces students to the computational methods used to process, analyze, and generate human language using modern machine learning and deep learning techniques. Combines foundational NLP concepts with practical implementation of neural network architectures used in contemporary language AI systems.

## Learning Objectives
- Process and represent text data (tokenization, stop-word removal, stemming, lemmatization, vectorization)
- Work with Sequence Models (RNNs, LSTMs) and neural networks (gradient optimization, feedforeard, embedding layers)
- Evaluate NLP systems (accuracy, precision, recall, F1 score)

## Folder Structure

```
W266-Natural-Language-Processing/
├── code/       # Scripts, notebooks, and source code
├── data/       # Raw and processed datasets
└─final_presentation/ #final deck   
```

## Final Project
An NLP research and engineering project where students design, build, evaluate, and present a modern natural language processing system using real-world text data and machine learning or deep learning methods.

We developed a model using knowledge graphs and graph convolutional networks (GCNs) that can classify diseases based on provided patient symptoms. Model applications include being able to treat patients more effectively, reduce room for error, and knowledge graphs offer a structured means of capturing relationships between entities.

First, we trained a BERT model by using splits of 80/10/10% for training, validation, and test datasets, respectively. Then, we constructed knwoedge graphs by mapping all diseases and associated symptoms as source and target nodes. Finally, we built the GCN, which involved text cleaning, making the knowledge graphs, then training the GCN. You can read more about our methodology and results in the final_presentation folder.

## Notes
Key libraries used: sklearn, tensorflow (tf, keras), seaborn, matplotlib

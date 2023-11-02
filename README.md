# Data Stream Mining 

This repository contains a data mining project that focuses on classifying data from streaming sources using a clustering-based ensemble classifier. The project utilizes the "Census Income" dataset to demonstrate the classification process. The core algorithm used is the GridDensity classifier, and ensemble learning techniques are employed to enhance classification performance.

## Project Overview

- **Problem Statement:** The project's primary goal is to classify data streams, which are continuously generated data sources, such as those collected by fast sensors. It is based on the concept of stream data mining.

- **Data Source:** The "Census Income" dataset is used for this project. This dataset contains various attributes related to individuals, and the goal is to predict whether an individual's income exceeds $50,000 per year based on these attributes.

- **Clustering-Based Classification:** The project employs a clustering-based approach to classify data points. It uses a clustering algorithm called GridDensity, which organizes data into grid cells based on their features and then assigns labels to those cells.

- **Ensemble Learning:** To enhance classification performance, the project utilizes ensemble learning. Specifically, a Bagging ensemble technique is applied, which combines multiple GridDensity models to make more robust predictions.

## Project Inspiration

This project is inspired by the research paper "[A clustering and ensemble based classifier for data stream classification](https://www.sciencedirect.com/science/article/abs/pii/S1568494620310140)," published in the "Applied Soft Computing Journal." The paper serves as the foundational source of the project's methodology and techniques.
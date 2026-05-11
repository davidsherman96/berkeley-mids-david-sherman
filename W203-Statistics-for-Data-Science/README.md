# W203: Statistics for Data Science

## Course Overview
Introduces students to classical statistical thinking and quantitative research methods within a modern data science context. Students learn how to formulate research questions, analyze uncertainty, draw valid references from data, and communicate statistical findings using the R programming language.

## Learning Objectives
- Apply foundational statistical concepts (descriptive/inferential stats, probability theory, confidence intervals, statistical significance)
- Conduct hypothesis testing: formulate null and alternate hypotheses, select appropriate stat tests, interpret p-values, evaluate Type I and Type II errors
- Build and interpret regression models
- Perform statistical analysis in R and communicate findings

## Folder Structure

```
W203-Statistics-for-Data-Science/
├── code/       # Scripts, notebooks, and source code
├── data/       # Raw and processed datasets
└── reports/    # Written reports, papers, and deliverables
```

## Final Project
A team-based applied statistical analysis project in which students investigate a real-world question using the statistical methods taught throughout the course.

For our project, we investigated the impact of a diamond's cut quality on its price. We aimed to identify which cuts maximize profitability - specifically, whether lower-tier cuts (like "Very Good") could be sold at similar prices to premium cuts, reducing costs while maintaining profit margins. Our methodology included:
- Dataset: ~27K diamonds from Kaggle's "Gemstone Price Prediction" (2015, USD)
- Data Split: 70% training, 30% test
- Approach: Three progressive linear regression models:
	- Cut only
	- Cut + carat
	- Cut + carat + color + clarity (primary model)
- Operationalization: cut quality (5 categories: Fair, Good, Very Good, Premium, Ideal) encoded as dummy variables with "Ideal" as the reference category

You can read more about our results and findings in our final report within the reports/ subfolder.

## Notes
Key R libraries used: tidyverse, ggplot2, GGally, sandwich, stargazer, rlang

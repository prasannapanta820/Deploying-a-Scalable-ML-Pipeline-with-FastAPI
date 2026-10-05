# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a scikit-learn `RandomForestClassifier` (scikit-learn 1.3.2, Python 3.8) built for the Udacity project "Deploying a Scalable ML Pipeline with FastAPI." It uses 100 trees with a maximum depth of 16, a minimum of 20 samples to split a node, and `random_state=42` so results are reproducible. Categorical features are one-hot encoded with a `OneHotEncoder` (unknown categories are ignored), and the label is binarized with a `LabelBinarizer`. The trained model and encoder are saved as `model/model.pkl` and `model/encoder.pkl` and served through a FastAPI application (`main.py`).

## Intended Use
The model predicts whether a person's annual income is above or below $50K based on demographic and employment attributes from the U.S. Census. It is intended for learning and demonstration purposes: practicing ML pipelines, slice-based evaluation, unit testing, CI, and API deployment. It should not be used to make real decisions about individuals, such as hiring, lending, insurance, or benefit eligibility.

## Training Data
The data is the UCI "Census Income" (Adult) dataset, extracted from the 1994 U.S. Census database (https://archive.ics.uci.edu/ml/datasets/census+income). The provided `census.csv` contains 32,561 rows and 15 columns: 6 numeric features (age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week), 8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country), and the label `salary` (`<=50K` or `>50K`). The classes are imbalanced: about 76% of records are `<=50K` and 24% are `>50K`. The data was split 80/20 into training and test sets with a stratified split on the label (`random_state=42`), giving 26,048 training rows. Missing values appear as `?` in some categorical columns and were kept as their own category.

## Evaluation Data
The evaluation data is the held-out 20% test split (6,513 rows), processed with the encoder and label binarizer fitted on the training data. In addition to overall metrics, the model was evaluated on every unique value of every categorical feature; those slice results are in `slice_output.txt`.

## Metrics
The model was evaluated with precision, recall, and F1 score, treating `>50K` as the positive class. On the test set the model achieved:

| Metric    | Value  |
|-----------|--------|
| Precision | 0.7957 |
| Recall    | 0.5963 |
| F1        | 0.6817 |

Precision is noticeably higher than recall, meaning that when the model predicts `>50K` it is usually right, but it misses about 40% of people who actually earn more than $50K.

Slice performance varies widely. F1 is similar for men (0.6821) and women (0.6798). Across race, F1 ranges from 0.6880 for White (5,533 rows) to 0.6053 for Black (662 rows) and 0.4615 for Amer-Indian-Eskimo (73 rows). By education, F1 is high for Masters (0.8806) and Prof-school (0.8715) but very low for HS-grad (0.3991), 10th (0.2353), and 7th-8th (0.0000). By occupation, Prof-specialty (0.8253) and Exec-managerial (0.8067) score well, while Other-service (0.2857) and Transport-moving (0.3117) score poorly. Slices with very few rows (for example, `workclass: Never-worked` with 1 row) report unstable metrics and should not be over-interpreted.

## Ethical Considerations
The dataset contains sensitive attributes such as race, sex, and native country, and the model uses them as inputs. Because the data reflects historical income patterns from 1994, the model can reproduce existing social and economic disparities. The slice analysis shows uneven performance across racial groups and education levels, so errors are not evenly distributed across the population. Any use that affects real people would require a fairness review, consideration of removing or controlling for protected attributes, and compliance with applicable anti-discrimination rules.

## Caveats and Recommendations
The data is more than 30 years old, and the $50K threshold is not adjusted for inflation, so predictions do not reflect today's income distribution. Some groups and categories have very small sample sizes, which makes their metrics unreliable. The class imbalance contributes to the low recall; future work could tune the decision threshold, use class weights, try gradient-boosted models, run hyperparameter search, or use K-fold cross-validation for more stable estimates. Slice metrics should be monitored whenever the model is retrained.

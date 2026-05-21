# Dynamic Seaborn Heatmap Report

This version generates a dynamic multi-page static HTML report from your CSV.

"Dynamic" here means **dynamic at generation time**:

- the script detects all learner families in the CSV
- it creates one page per learner family
- it creates the navbar automatically
- it creates the home overview heatmap automatically

The final result is still static HTML, so it is easy to open, archive, and submit.

## Output pages

Example output:

```text
reports/metrics_report/
├── index.html
├── decision-tree.html
├── logistic-regression.html
├── random-forest.html
├── svm.html
├── gradient-boosting.html
└── neural-network.html
```

Only pages for learner families present in the CSV are generated.

## Install dependencies

Using `uv`:

```bash
uv add pandas seaborn matplotlib jinja2
```

## Expected CSV

The script expects a learner/model column such as:

```text
Learner
```

and numeric metric columns, for example:

```text
CA
AUC
Precision(average=macro)
Recall(average=macro)
F1(average=macro)
MCC
```

## Run

From the project root:

```bash
python src/generate_report.py \
  --csv results/evaluation-results.csv \
  --templates-dir templates \
  --output-dir reports/metrics_report \
  --top-n 30
```

Then open:

```text
reports/metrics_report/index.html
```

## What the home page shows

The home page shows only:

```text
Best configuration per learner family
```

This is the overview heatmap.

## What each family page shows

Each learner-family page shows:

```text
Top N configurations for that learner family
Configuration tracking table
Full family results table
```

## How grouping works

The script derives the learner family from the learner string:

```text
LogisticRegressionLearner(C=1) -> LogisticRegressionLearner
RandomForestLearner(n_estimators=300) -> RandomForestLearner
TreeLearner(max_depth=5) -> TreeLearner
```

Then it maps technical names to readable labels:

```text
LogisticRegressionLearner -> Logistic Regression
RandomForestLearner -> Random Forest
TreeLearner -> Decision Tree
```

Unknown learner names are still supported. They are slugified and receive their own generated page.

## Tracking

The report creates short labels for heatmaps:

```text
Random Forest #001
Random Forest #002
SVM #001
```

The tracking table maps those labels back to the full learner configuration.

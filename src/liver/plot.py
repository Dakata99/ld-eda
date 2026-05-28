from pathlib import Path
import re
import shutil

from jinja2 import Environment, FileSystemLoader
from loguru import logger
import pandas as pd
import plotly.graph_objects as go

from .utils import root

TEMPLATES: Path = root("templates")

LEARNER_TO_FAMILY_MAPPING: dict[str, str] = {
	"LogisticRegressionLearner": "LR",
	"RandomForestLearner": "RF",
	"TreeLearner": "DT",
	"GBClassifier": "GB",
	"NNClassificationLearner": "NN",
	"SVMLearner": "SVM",
}

FAMILY_TO_FILE_MAPPING: dict[str, str] = {
	"LR": "logistic-regression",
	"DT": "tree",
	"RF": "random-forest",
	"GB": "gradient-boosting",
	"NN": "neural-network",
	"SVM": "svm",
}

HEATMAPS: dict = {
	"all": None,
	"per-family": None
}

TOP_6: int = 6
TOP_20: int = 20
TOP_N: int = TOP_20

INDEX_TEMPLATE: str = "index.html"
LEARNER_TEMPLATE: str = "learner.html"


# NOTE: class_weight and random_state are hardcoded in the JSON files,
# so they are known!
def simplify_learner_repr(s):
    # Remove ', class_weight=...' or 'class_weight=...,'
    s = re.sub(r", class_weight='balanced'", "", s)
    s = re.sub(r"class_weight='balanced', ", "", s)

    # Remove ', random_state=...' or 'random_state=...,'
    s = re.sub(r", random_state=[^,)]*", "", s)
    s = re.sub(r"random_state=[^,)]*, ", "", s)

    # Remove 'min_impurity_split=<?>'
    s = re.sub(r",?\s*min_impurity_split=[^,)]*", "", s)

    # Remove 'presort=<?>'
    s = re.sub(r",?\s*presort=[^,)]*", "", s)

    return s.strip()


def rename_learner(s: str):
	name, rest = s.split("(", 1)
	return f'{LEARNER_TO_FAMILY_MAPPING[name]}({rest}'


def heatmap(df: pd.DataFrame):
	df = df.set_index("ID")
	columns = df.drop(columns=["Learner", "Family"]).columns.tolist()
	values = df.select_dtypes(include="number").to_numpy()
	fig = go.Figure(
		data=[
			go.Heatmap(
				z=values,
				x=columns,
				y=df.index.tolist(),
				colorscale="Viridis",
				text=[[f"{value:.8f}" for value in row] for row in values],
				texttemplate="%{text}",
				hoverinfo="none",
				colorbar={"title": "Value"},
			)
		]
	)
	fig.update_layout(
		xaxis_title="Metric",
		yaxis_title="Learner",
		autosize=True,
		margin={"l": 120, "r": 20, "t": 60, "b": 80},
		height=600,
	)

	return fig.to_html(
		full_html=False,
		include_plotlyjs="cdn",
	)


def main(exprid: int, method: str, config: str):
	# 1) Load the results (CSV file) into a DataFrame
	fd = root("results", f"experiment{exprid}-{config}-{method}.csv")
	if not fd.exists():
		raise FileNotFoundError(f"Results file not found: {fd}")

	df = pd.read_csv(fd)
	logger.success(f"Loaded file: {fd}")

	# FIXME: drop weighted matrics for scenario 2 and 3
	if any('Sick' in column for column in df.columns):
		df.drop(columns=['Recall(weighted)', 'F1(weighted)'], inplace=True)

	# Simplify learner representation
	df["Learner"] = df['Learner'].apply(rename_learner)
	df['Learner'] = df['Learner'].apply(simplify_learner_repr)
	logger.debug(df.head())

	# Priority is preserved by the ordering of the metrics in the CSV file!
	metrics = df.columns.to_list()
	metrics.pop(0)  # remove 'Learner' column
	rows = len(df)

	df = df.sort_values(by=metrics, ascending=[False] * len(metrics))
	logger.debug(df.head())

	# Map learner names
	df["Family"] = df["Learner"].apply(lambda learner: learner.split("(", 1)[0])
	df["ID"] = df["Family"] + "#" + (df.groupby("Family").cumcount() + 1).astype(str).str.zfill(3)
	logger.debug(df.head())

	families = df["Family"].unique()
	for family in families:
		famdf = df[df["Family"] == family]
		HEATMAPS[family] = heatmap(famdf.head(TOP_N))

	# 2) Generate heatmap
	HEATMAPS["all"] = heatmap(df.head(TOP_N))
	top_per_family = df.groupby('Family', sort=False).head(1)
	HEATMAPS["per-family"] = heatmap(top_per_family)

    # 3) Create HTML reports
	env = Environment(loader=FileSystemLoader(TEMPLATES))
	template = env.get_template(INDEX_TEMPLATE)

	# Create navigation items
	nav_items = [
		{
			"key": "home",
			"label": "Home",
			"file": "index.html",
		}
	]

	for family in families:
		nav_items.append(
			{
				"key": family,
				"label": FAMILY_TO_FILE_MAPPING[family],
				"file": f"{FAMILY_TO_FILE_MAPPING[family]}.html",
			}
		)

	index = template.render(
		# <head>
		page_title="Metrics Report — Overview",
		active_page="home",
		# Navigation bar
		nav_items=nav_items,
		# Hero card
		report_title=f"Experiment {exprid} overview",
		# Summary grid
		total_rows=rows,
		family_count=len(df["Family"].unique()),
		num_metrics=len(metrics),
		metrics_priority="<br>".join(metrics),
		# Heatmap cards
		heatmap_all=HEATMAPS['per-family'],
		top_n=TOP_N,
		heatmap=HEATMAPS["all"],
		# Evaluation results card
		evaluation_results=df.drop(columns=["Family"]).to_html(index=False, table_id="results-table"),
	)

	expr = root("reports", f"expr{exprid}-{config}-{method}")
	if not expr.exists():
		expr.mkdir(parents=True)

	output_file: Path = root("reports", expr / 'index.html')
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(index, encoding="utf-8")

	logger.success(f"HTML report generated: {output_file}")

	learner_template = env.get_template(LEARNER_TEMPLATE)
	for family in families:
		famdf = df[df["Family"] == family]
		page = learner_template.render(
			# <head>
			page_title="Metrics Report — Overview",
			active_page=family,
			# Navigation bar
			nav_items=nav_items,
			# Hero card
			report_title=FAMILY_TO_FILE_MAPPING[family],
			# Summary grid
			total_rows=len(famdf),
			num_metrics=len(metrics),
			metrics_priority="<br>".join(metrics),
			# Heatmap card
			top_n=TOP_N,
			heatmap=HEATMAPS[family],
			# Evaluation results card
			evaluation_results=famdf.drop(columns=["Family"]).to_html(index=False, table_id="results-table"),
		)

		ofile: Path = root("reports", expr / f"{FAMILY_TO_FILE_MAPPING[family]}.html")
		ofile.parent.mkdir(parents=True, exist_ok=True)
		ofile.write_text(page, encoding="utf-8")

		logger.success(f"HTML report generated: {ofile}")

	# Copy CSS file next to generated files
	shutil.copy(root("templates/styles.css"), expr)
	shutil.copy(root("templates/script.js"), expr)

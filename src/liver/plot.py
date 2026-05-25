from pathlib import Path
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

HEATMAPS: dict = {"overview": None}

TOP_20: int = 20

INDEX_TEMPLATE: str = "index.html"
LEARNER_TEMPLATE: str = "learner.html"


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


def main(exprid: int, config: str, filename: str):
	# 1) Load the results (CSV file) into a DataFrame
	fd = root("results", filename)
	if not fd.exists():
		raise FileNotFoundError(f"Results file not found: {fd}")

	df = pd.read_csv(fd)
	logger.success(f"Loaded file: {fd}")
	logger.debug(df.head())

	# Priority is preserved by the ordering of the metrics in the CSV file!
	metrics = list(df.columns)
	metrics.pop(0)  # remove 'Learner' column
	rows = len(df)

	df = df.sort_values(by=metrics, ascending=[False] * len(metrics))
	logger.debug(df.head())

	# Map learner names
	df["Family"] = df["Learner"].apply(lambda learner: learner.split("(")[0])
	df["Family"] = df["Family"].map(LEARNER_TO_FAMILY_MAPPING)
	df["ID"] = df["Family"] + "#" + (df.groupby("Family").cumcount() + 1).astype(str).str.zfill(3)
	logger.debug(df.head())

	families = df["Family"].unique()
	for family in families:
		famdf = df[df["Family"] == family]
		# famdf = famdf.sort_values(by=metrics, ascending=[False] * len(metrics))
		HEATMAPS[family] = heatmap(famdf)

	# 2) Generate heatmap
	HEATMAPS["overview"] = heatmap(df.head(TOP_20))

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
		# Heatmap card
		top_n=TOP_20,
		heatmap=HEATMAPS["overview"],
		# Evaluation results card
		evaluation_results=df.to_html(index=False, table_id="results-table"),
	)

	expr = root("reports", f"expr{exprid}-{config}")
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
			top_n=TOP_20,
			heatmap=HEATMAPS[family],
			# Evaluation results card
			evaluation_results=famdf.to_html(index=False, table_id="results-table"),
		)

		ofile: Path = root("reports", expr / f"{FAMILY_TO_FILE_MAPPING[family]}.html")
		ofile.parent.mkdir(parents=True, exist_ok=True)
		ofile.write_text(page, encoding="utf-8")

		logger.success(f"HTML report generated: {ofile}")

	# Copy CSS file next to generated files
	shutil.copy(root("templates/styles.css"), expr)
	shutil.copy(root("templates/script.js"), expr)

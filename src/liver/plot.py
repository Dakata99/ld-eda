from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger
import pandas as pd
import plotly.graph_objects as go

from .utils import root

TEMPLATES: Path = root("templates")


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


def main(exprid: int, filename: str, outputfile: str):
	# 1) Load the results (CSV file)
	fd = root("results", filename)
	if not fd.exists():
		raise FileNotFoundError(f"Results file not found: {fd}")

	# Load the CSV file into a DataFrame
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
	mapping: dict[str, str] = {
		"LogisticRegressionLearner": "LR",
		"RandomForestLearner": "RF",
		"TreeLearner": "TR",
		"GBClassifier": "GB",
		"NNClassificationLearner": "NN",
		"SVMLearner": "SVM",
	}
	df["Family"] = df["Learner"].apply(lambda learner: learner.split("(")[0])
	df["Family"] = df["Family"].map(mapping)
	df["ID"] = df["Family"] + "#" + (df.groupby("Family").cumcount() + 1).astype(str).str.zfill(3)
	logger.debug(df)

	# 2) Generate heatmap
	# TODO: per-learner family heat map
	# TODO: best per family learner heat map
	hmap = heatmap(df.head(20))
	table = df.to_html(
		index=False, table_id="results-table", classes="table table-striped table-bordered"
	)
	full_table = f"""
		<div style="overflow-x: auto; width: 100%; margin: 0;">
			{table}
		</div>
	"""

	env = Environment(loader=FileSystemLoader(TEMPLATES))
	template = env.get_template("report.html")

	html = template.render(
		# <head>
		page_title="Metrics Report — Overview",
		active_page="home",
		# Hero card
		report_title=f"Experiment {exprid} report overview",
		# Summary grid
		total_rows=rows,
		family_count=len(df["Family"].unique()),
		num_metrics=len(metrics),
		metrics_priority="<br>".join(metrics),
		# Heatmap card
		heatmap=hmap,
		# Evaluation results card
		evaluation_results=full_table,
	)

	output_file: Path = root("reports", outputfile)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(html, encoding="utf-8")

	logger.success(f"HTML report generated: {output_file}")

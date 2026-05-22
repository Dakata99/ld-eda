from __future__ import annotations

import argparse
import base64
from collections.abc import Iterable
import io
from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FAMILY_LABELS = {
	"LogisticRegressionLearner": "Logistic Regression",
	"RandomForestLearner": "Random Forest",
	"SVMLearner": "SVM",
	"SklSVMLearner": "SVM",
	"TreeLearner": "Decision Tree",
	"GBClassifier": "Gradient Boosting",
	"GradientBoostingLearner": "Gradient Boosting",
	"NNClassificationLearner": "Neural Network",
	"MLPLearner": "Neural Network",
}

FAMILY_KEYS = {
	"LogisticRegressionLearner": "logistic-regression",
	"RandomForestLearner": "random-forest",
	"SVMLearner": "svm",
	"SklSVMLearner": "svm",
	"TreeLearner": "decision-tree",
	"GBClassifier": "gradient-boosting",
	"GradientBoostingLearner": "gradient-boosting",
	"NNClassificationLearner": "neural-network",
	"MLPLearner": "neural-network",
}

MAIN_METRIC_PRIORITY = [
	"recall_sick",
	"Recall(Sick)",
	"Recall(class=Sick)",
	"f1_sick",
	"F1(Sick)",
	"F1(class=Sick)",
	"F1(average=macro)",
	"Recall(average=macro)",
	"MCC",
	"balanced_accuracy",
	"Balanced Accuracy",
	"AUC",
	"CA",
	"accuracy",
]

NON_METRIC_COLUMNS = {
	"config_index",
	"fold",
	"seed",
	"random_seed",
	"n_estimators",
	"max_depth",
	"min_samples_leaf",
	"min_samples_split",
	"max_iter",
	"degree",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate a dynamic multi-page Seaborn heatmap HTML report from a CSV file."
	)

	parser.add_argument(
		"--csv",
		required=True,
		type=Path,
		help="Path to the CSV file with evaluation results.",
	)

	parser.add_argument(
		"--output-dir",
		default=Path("reports/metrics_report"),
		type=Path,
		help="Directory where the generated HTML files will be saved.",
	)

	parser.add_argument(
		"--templates-dir",
		default=Path("templates"),
		type=Path,
		help="Directory containing the Jinja2 HTML templates.",
	)

	parser.add_argument(
		"--top-n",
		default=30,
		type=int,
		help="Maximum number of configurations shown in each family heatmap.",
	)

	return parser.parse_args()


def slugify(value: str) -> str:
	value = value.lower()
	value = re.sub(r"[^a-z0-9]+", "-", value)
	return value.strip("-")


def derive_family_name(learner_value: object) -> str:
	value = str(learner_value).strip()

	if "(" in value:
		return value.split("(", 1)[0].strip()

	return value


def detect_learner_column(df: pd.DataFrame) -> str:
	candidates = [
		"Learner",
		"learner",
		"model",
		"model_label",
		"configuration",
	]

	for candidate in candidates:
		if candidate in df.columns:
			return candidate

	raise ValueError("Could not detect learner column. Expected one of: " + ", ".join(candidates))


def detect_metric_columns(df: pd.DataFrame) -> list[str]:
	metrics: list[str] = []

	for column in df.select_dtypes(include="number").columns:
		normalized = column.lower()

		if normalized in NON_METRIC_COLUMNS:
			continue

		metrics.append(column)

	if not metrics:
		raise ValueError("No numeric metric columns were detected.")

	return metrics


def choose_main_metric(metrics: Iterable[str]) -> str:
	metrics = list(metrics)

	for preferred_metric in MAIN_METRIC_PRIORITY:
		if preferred_metric in metrics:
			return preferred_metric

	return metrics[0]


def add_tracking_columns(df: pd.DataFrame, learner_column: str) -> pd.DataFrame:
	df = df.copy()

	df["learner_family_raw"] = df[learner_column].apply(derive_family_name)

	df["learner_family_label"] = (
		df["learner_family_raw"].map(FAMILY_LABELS).fillna(df["learner_family_raw"])
	)

	df["learner_family_key"] = (
		df["learner_family_raw"].map(FAMILY_KEYS).fillna(df["learner_family_label"].apply(slugify))
	)

	df["learner_family_file"] = df["learner_family_key"] + ".html"

	df["config_index"] = df.groupby("learner_family_key").cumcount() + 1

	df["config_label"] = (
		df["learner_family_label"] + " #" + df["config_index"].astype(str).str.zfill(3)
	)

	return df


def dataframe_to_html_table(df: pd.DataFrame) -> str:
	return df.to_html(index=False, classes="table", border=0, escape=False)


def figure_to_base64_img(fig: plt.Figure, alt: str) -> str:
	buffer = io.BytesIO()

	fig.savefig(
		buffer,
		format="png",
		dpi=180,
		bbox_inches="tight",
		facecolor="white",
	)

	plt.close(fig)
	buffer.seek(0)

	encoded = base64.b64encode(buffer.read()).decode("utf-8")

	return f'<img class="heatmap-img" src="data:image/png;base64,{encoded}" alt="{alt}" />'


def create_heatmap(
	df: pd.DataFrame,
	metrics: list[str],
	label_column: str,
	title: str,
) -> str:
	if df.empty:
		return '<div class="notice">No rows available for this heatmap.</div>'

	matrix = df.set_index(label_column)[metrics]

	height = max(4.5, len(matrix) * 0.42)
	width = max(9.5, len(metrics) * 1.35)

	fig, ax = plt.subplots(figsize=(width, height))

	sns.heatmap(
		matrix,
		annot=True,
		fmt=".3f",
		linewidths=0.5,
		cbar=True,
		ax=ax,
	)

	ax.set_title(title, fontsize=14, pad=16)
	ax.set_xlabel("Metrics")
	ax.set_ylabel("Configuration")

	plt.xticks(rotation=35, ha="right")
	plt.yticks(rotation=0)

	fig.tight_layout()

	return figure_to_base64_img(fig, alt=title)


def create_overview_df(df: pd.DataFrame, main_metric: str) -> pd.DataFrame:
	best_indices = df.groupby("learner_family_key")[main_metric].idxmax()

	return df.loc[best_indices].sort_values(main_metric, ascending=False).reset_index(drop=True)


def create_tracking_table(
	df: pd.DataFrame,
	learner_column: str,
	metrics: list[str],
) -> str:
	columns = [
		"config_label",
		"learner_family_label",
		learner_column,
		*metrics,
	]

	existing_columns = [column for column in columns if column in df.columns]

	return dataframe_to_html_table(df[existing_columns])


def build_family_metadata(
	df: pd.DataFrame,
	main_metric: str,
) -> list[dict[str, object]]:
	metadata: list[dict[str, object]] = []

	grouped = df.groupby("learner_family_key", sort=False)

	for family_key, family_df in grouped:
		family_df = family_df.sort_values(main_metric, ascending=False)

		first = family_df.iloc[0]

		metadata.append(
			{
				"key": family_key,
				"label": first["learner_family_label"],
				"file": first["learner_family_file"],
				"count": len(family_df),
				"best_score": f"{first[main_metric]:.4f}",
			}
		)

	return sorted(metadata, key=lambda item: str(item["label"]))


def build_nav_items(family_metadata: list[dict[str, object]]) -> list[dict[str, str]]:
	nav_items = [
		{
			"key": "home",
			"label": "Home",
			"file": "index.html",
		}
	]

	for family in family_metadata:
		nav_items.append(
			{
				"key": str(family["key"]),
				"label": str(family["label"]),
				"file": str(family["file"]),
			}
		)

	return nav_items


def render_home_page(
	env: Environment,
	df: pd.DataFrame,
	overview_df: pd.DataFrame,
	learner_column: str,
	metrics: list[str],
	main_metric: str,
	family_metadata: list[dict[str, object]],
	nav_items: list[dict[str, str]],
	output_dir: Path,
) -> None:
	template = env.get_template("index.html")

	overview_heatmap = create_heatmap(
		df=overview_df,
		metrics=metrics,
		label_column="config_label",
		title="Overview: Best Configuration per Learner Family",
	)

	overview_tracking_table = create_tracking_table(
		df=overview_df,
		learner_column=learner_column,
		metrics=metrics,
	)

	html = template.render(
		page_title="Metrics Report — Overview",
		active_page="home",
		nav_items=nav_items,
		report_title="Liver Disease Classification — Metrics Overview",
		total_rows=len(df),
		family_count=df["learner_family_key"].nunique(),
		metric_count=len(metrics),
		main_metric=main_metric,
		family_cards=family_metadata,
		overview_heatmap=overview_heatmap,
		overview_tracking_table=overview_tracking_table,
	)

	(output_dir / "index.html").write_text(html, encoding="utf-8")


def render_family_page(
	env: Environment,
	df: pd.DataFrame,
	learner_column: str,
	metrics: list[str],
	main_metric: str,
	family: dict[str, object],
	nav_items: list[dict[str, str]],
	top_n: int,
	output_dir: Path,
) -> None:
	template = env.get_template("family.html")

	family_key = str(family["key"])
	family_label = str(family["label"])
	output_file = str(family["file"])

	family_df = (
		df[df["learner_family_key"] == family_key]
		.sort_values(main_metric, ascending=False)
		.reset_index(drop=True)
	)

	top_df = family_df.head(top_n)

	best_score = "N/A" if family_df.empty else f"{family_df.iloc[0][main_metric]:.4f}"

	family_heatmap = create_heatmap(
		df=top_df,
		metrics=metrics,
		label_column="config_label",
		title=f"{family_label}: Top {min(top_n, len(family_df))} Configurations",
	)

	tracking_table = create_tracking_table(
		df=family_df,
		learner_column=learner_column,
		metrics=metrics,
	)

	family_results_table = dataframe_to_html_table(family_df)

	html = template.render(
		page_title=f"Metrics Report — {family_label}",
		active_page=family_key,
		nav_items=nav_items,
		family_label=family_label,
		family_total_rows=len(family_df),
		shown_rows=len(top_df),
		top_n=top_n,
		main_metric=main_metric,
		best_score=best_score,
		family_heatmap=family_heatmap,
		tracking_table=tracking_table,
		family_results_table=family_results_table,
	)

	(output_dir / output_file).write_text(html, encoding="utf-8")


def main() -> None:
	args = parse_args()

	df = pd.read_csv(args.csv)
	learner_column = detect_learner_column(df)
	metrics = detect_metric_columns(df)
	main_metric = choose_main_metric(metrics)

	df = add_tracking_columns(df, learner_column)

	overview_df = create_overview_df(df, main_metric)
	family_metadata = build_family_metadata(df, main_metric)
	nav_items = build_nav_items(family_metadata)

	args.output_dir.mkdir(parents=True, exist_ok=True)

	env = Environment(loader=FileSystemLoader(args.templates_dir))

	render_home_page(
		env=env,
		df=df,
		overview_df=overview_df,
		learner_column=learner_column,
		metrics=metrics,
		main_metric=main_metric,
		family_metadata=family_metadata,
		nav_items=nav_items,
		output_dir=args.output_dir,
	)

	for family in family_metadata:
		render_family_page(
			env=env,
			df=df,
			learner_column=learner_column,
			metrics=metrics,
			main_metric=main_metric,
			family=family,
			nav_items=nav_items,
			top_n=args.top_n,
			output_dir=args.output_dir,
		)

	print(f"CSV: {args.csv}")
	print(f"Learner column: {learner_column}")
	print(f"Detected metrics: {metrics}")
	print(f"Main metric: {main_metric}")
	print("Detected learner families:")
	for family in family_metadata:
		print(f"  - {family['label']} ({family['count']} configurations) -> {family['file']}")
	print(f"Report generated in: {args.output_dir}")


if __name__ == "__main__":
	main()

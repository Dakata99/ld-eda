#!/usr/bin/env python3
"""
Interactive multidimensional report for learner configurations.

Highlights
----------
- Uses SAMPLE_RESULTS directly, so it runs out of the box.
- Builds grouped configuration IDs like LR1, LR2, RF1, RF2, ...
- Shows full parameter settings in hover tooltips and in ranking tables.
- Generates a self-contained interactive HTML report.
- Lets you choose the primary metric dynamically inside the report.
- Lets you toggle raw vs normalized chart values.
- Includes built-in help panels on how to interpret the visuals.

Dependencies
------------
pip install pandas numpy plotly

Run
---
python learner_multidim_interactive_report.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import seaborn as sns
import matplotlib.pyplot as plt


SAMPLE_RESULTS = [
	{
		"learner": "logistic-regression",
		"params": {"C": 0.1, "penalty": "l2"},
		"MCC": 0.71,
		"F1": 0.78,
		"AUC": 0.85,
		"CA": 0.80,
	},
	{
		"learner": "logistic-regression",
		"params": {"C": 1.0, "penalty": "l2"},
		"MCC": 0.76,
		"F1": 0.82,
		"AUC": 0.88,
		"CA": 0.83,
	},
	{
		"learner": "logistic-regression",
		"params": {"C": 5.0, "penalty": "l1"},
		"MCC": 0.74,
		"F1": 0.80,
		"AUC": 0.87,
		"CA": 0.82,
	},
	{
		"learner": "random-forest",
		"params": {"n_estimators": 100, "max_depth": None},
		"MCC": 0.82,
		"F1": 0.87,
		"AUC": 0.93,
		"CA": 0.88,
	},
	{
		"learner": "random-forest",
		"params": {"n_estimators": 300, "max_depth": 10},
		"MCC": 0.85,
		"F1": 0.89,
		"AUC": 0.95,
		"CA": 0.90,
	},
	{
		"learner": "random-forest",
		"params": {"n_estimators": 100, "max_depth": 10},
		"MCC": 0.83,
		"F1": 0.88,
		"AUC": 0.94,
		"CA": 0.89,
	},
	{
		"learner": "svm",
		"params": {"C": 0.1, "kernel": "rbf"},
		"MCC": 0.68,
		"F1": 0.74,
		"AUC": 0.81,
		"CA": 0.77,
	},
	{
		"learner": "svm",
		"params": {"C": 1.0, "kernel": "rbf"},
		"MCC": 0.79,
		"F1": 0.84,
		"AUC": 0.90,
		"CA": 0.85,
	},
	{
		"learner": "svm",
		"params": {"C": 10, "kernel": "linear"},
		"MCC": 0.77,
		"F1": 0.83,
		"AUC": 0.89,
		"CA": 0.84,
	},
	{
		"learner": "gradient-boosting",
		"params": {"n_estimators": 100, "lr": 0.1},
		"MCC": 0.86,
		"F1": 0.90,
		"AUC": 0.96,
		"CA": 0.91,
	},
	{
		"learner": "gradient-boosting",
		"params": {"n_estimators": 300, "lr": 0.05},
		"MCC": 0.88,
		"F1": 0.92,
		"AUC": 0.97,
		"CA": 0.93,
	},
	{
		"learner": "neural-network",
		"params": {"hidden": [100], "lr": 0.001},
		"MCC": 0.80,
		"F1": 0.85,
		"AUC": 0.92,
		"CA": 0.87,
	},
	{
		"learner": "neural-network",
		"params": {"hidden": [50, 50], "lr": 0.0001},
		"MCC": 0.78,
		"F1": 0.83,
		"AUC": 0.91,
		"CA": 0.86,
	},
]

OUTPUT_DIR = Path(__file__).resolve().parent / "learner_multidim_interactive_output"
DEFAULT_PRIMARY_METRIC = "MCC"
DEFAULT_NORMALIZE = True

PREFERRED_METRIC_ORDER = [
	"MCC",
	"PR_AUC",
	"F1",
	"Recall",
	"Precision",
	"Balanced_Accuracy",
	"CA",
	"AUC",
	"ROC_AUC",
]

PREFERRED_LEARNER_ORDER = [
	"logistic-regression",
	"random-forest",
	"svm",
	"gradient-boosting",
	"neural-network",
]


def is_number(value: Any) -> bool:
	return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def detect_metric_columns(results: Sequence[Dict[str, Any]]) -> List[str]:
	ignore = {
		"learner",
		"params",
		"config",
		"config_id",
		"learner_family",
		"learner_abbr",
	}
	discovered = []
	seen = set()

	for row in results:
		for key, value in row.items():
			if key in ignore:
				continue
			if is_number(value) and key not in seen:
				discovered.append(key)
				seen.add(key)

	preferred_first = [m for m in PREFERRED_METRIC_ORDER if m in seen]
	remaining = [m for m in discovered if m not in preferred_first]
	return preferred_first + remaining


def learner_abbreviation(learner_name: str) -> str:
	explicit = {
		"logistic-regression": "LR",
		"random-forest": "RF",
		"svm": "SVM",
		"gradient-boosting": "GB",
		"neural-network": "NN",
		"decision-tree": "DT",
		"tree": "DT",
		"naive-bayes": "NB",
		"knn": "KNN",
	}
	if learner_name in explicit:
		return explicit[learner_name]

	parts = [p for p in learner_name.replace("_", "-").split("-") if p]
	if not parts:
		return "MODEL"
	if len(parts) == 1:
		return parts[0][:3].upper()
	return "".join(p[0].upper() for p in parts)


def stable_json(value: Any) -> str:
	return json.dumps(value, sort_keys=True, ensure_ascii=False)


def pretty_json(value: Any) -> str:
	return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)


def get_learner_order(results: Sequence[Dict[str, Any]]) -> List[str]:
	seen = []
	for row in results:
		learner = row["Learner"]
		if learner not in seen:
			seen.append(learner)
	ordered = [x for x in PREFERRED_LEARNER_ORDER if x in seen]
	ordered += [x for x in seen if x not in ordered]
	return ordered


def metric_theoretical_bounds(metric_name: str) -> tuple[float, float]:
	metric_upper = metric_name.upper()
	if metric_upper == "MCC":
		return -1.0, 1.0

	bounded_0_1_names = {
		"F1",
		"AUC",
		"ROC_AUC",
		"PR_AUC",
		"CA",
		"PRECISION",
		"RECALL",
		"BALANCED_ACCURACY",
		"SPECIFICITY",
		"SENSITIVITY",
	}
	if metric_upper in bounded_0_1_names:
		return 0.0, 1.0

	return math.nan, math.nan


def normalize_series(series: pd.Series, metric_name: str) -> pd.Series:
	lower, upper = metric_theoretical_bounds(metric_name)

	if math.isnan(lower) or math.isnan(upper):
		observed_min = series.min()
		observed_max = series.max()
		if pd.isna(observed_min) or pd.isna(observed_max) or observed_min == observed_max:
			return pd.Series(np.ones(len(series)), index=series.index)
		return (series - observed_min) / (observed_max - observed_min)

	normalized = (series - lower) / (upper - lower)
	return normalized.clip(0.0, 1.0)


def build_dataframe(
	results: Sequence[Dict[str, Any]],
) -> tuple[pd.DataFrame, List[str]]:
	metrics = detect_metric_columns(results)
	learner_order = get_learner_order(results)
	learner_order_index = {name: i for i, name in enumerate(learner_order)}

	counters = defaultdict(int)
	rows = []

	for raw in results:
		learner = raw["Learner"]
		counters[learner] += 1
		config_index = counters[learner]
		abbr = learner_abbreviation(learner)
		config_id = f"{abbr}{config_index}"

		row = {
			"learner": learner,
			"learner_abbr": abbr,
			"config_index": config_index,
			"config_id": config_id,
			"params": raw.get("params", {}),
			"params_json": stable_json(raw.get("params", {})),
			"params_pretty": pretty_json(raw.get("params", {})),
			"learner_order": learner_order_index[learner],
		}
		for metric in metrics:
			row[metric] = raw.get(metric, np.nan)
		rows.append(row)

	df = (
		pd.DataFrame(rows)
		.sort_values(by=["learner_order", "config_index"], kind="stable")
		.reset_index(drop=True)
	)

	for metric in metrics:
		df[f"{metric}__norm"] = normalize_series(df[metric], metric)

	return df, metrics


def metric_explanation_block(metrics: Sequence[str]) -> str:
	bullets = []
	for metric in metrics:
		if metric.upper() == "MCC":
			bullets.append(
				"<li><strong>MCC</strong>: strong single-number summary, especially useful for imbalanced classification. Closer to 1 is better.</li>"
			)
		elif metric.upper() == "F1":
			bullets.append(
				"<li><strong>F1</strong>: balance between precision and recall. Higher is better.</li>"
			)
		elif metric.upper() in {"AUC", "ROC_AUC"}:
			bullets.append(
				"<li><strong>AUC / ROC AUC</strong>: ranking quality across thresholds. Higher is better.</li>"
			)
		elif metric.upper() == "CA":
			bullets.append(
				"<li><strong>CA</strong>: classification accuracy. Easy to read, but can be misleading for imbalanced data.</li>"
			)
		elif metric.upper() == "PR_AUC":
			bullets.append(
				"<li><strong>PR AUC</strong>: often more informative than ROC AUC when the positive class is rare. Higher is better.</li>"
			)
		elif metric.upper() == "RECALL":
			bullets.append(
				"<li><strong>Recall</strong>: how many actual positives were found. Higher is better when missing positives is costly.</li>"
			)
		elif metric.upper() == "PRECISION":
			bullets.append(
				"<li><strong>Precision</strong>: how many predicted positives were actually correct. Higher is better when false alarms are costly.</li>"
			)
		else:
			bullets.append(f"<li><strong>{metric}</strong>: higher is assumed to be better.</li>")
	return "\n".join(bullets)


def make_heatmap_figure(
	df: pd.DataFrame, metrics: Sequence[str], normalize: bool, primary_metric: str
) -> go.Figure:
	working = (
		df.copy()
		.sort_values(by=["learner_order", primary_metric], ascending=[True, False], kind="stable")
		.reset_index(drop=True)
	)

	z = []
	raw_text = []
	customdata = []
	y_labels = []

	for _, row in working.iterrows():
		y_labels.append(row["config_id"])
		z_row = []
		raw_row = []
		custom_row = []
		for metric in metrics:
			raw_value = float(row[metric])
			plot_value = float(row[f"{metric}__norm"]) if normalize else raw_value
			z_row.append(plot_value)
			raw_row.append(f"{raw_value:.3f}")
			custom_row.append(
				[
					row["config_id"],
					row["learner"],
					row["params_pretty"],
					metric,
					raw_value,
					float(row[f"{metric}__norm"]),
				]
			)
		z.append(z_row)
		raw_text.append(raw_row)
		customdata.append(custom_row)

	fig = go.Figure(
		data=[
			go.Heatmap(
				z=z,
				x=list(metrics),
				y=y_labels,
				customdata=customdata,
				text=raw_text,
				texttemplate="%{text}",
				colorscale="Viridis",
				colorbar={"title": "Normalized" if normalize else "Raw"},
				hovertemplate=(
					"<b>%{customdata[0]}</b><br>"
					"Learner: %{customdata[1]}<br>"
					"Metric: %{customdata[3]}<br>"
					"Raw value: %{customdata[4]:.4f}<br>"
					"Normalized: %{customdata[5]:.4f}<br>"
					"<br><b>Parameters</b><br><span style='font-family:monospace'>%{customdata[2]}</span>"
					"<extra></extra>"
				),
			)
		]
	)
	fig.update_layout(
		title=f"Heatmap overview — grouped by learner family, sorted within family by {primary_metric}",
		xaxis_title="Metrics",
		yaxis_title="Configurations",
		height=max(550, 40 * len(working) + 220),
		margin=dict(l=90, r=20, t=80, b=70),
	)
	return fig


def make_profile_figure(
	df: pd.DataFrame, metrics: Sequence[str], normalize: bool, primary_metric: str
) -> go.Figure:
	working = (
		df.copy()
		.sort_values(by=["learner_order", primary_metric], ascending=[True, False], kind="stable")
		.reset_index(drop=True)
	)
	fig = go.Figure()
	families = working["learner"].unique().tolist()
	traces_per_family = []

	for family in families:
		sub = working[working["learner"] == family].copy()
		family_trace_indices = []
		for _, row in sub.iterrows():
			y_values = [
				float(row[f"{metric}__norm"]) if normalize else float(row[metric])
				for metric in metrics
			]
			hover_parts = [
				f"<b>{row['config_id']}</b>",
				f"Learner: {row['learner']}",
				f"Primary metric ({primary_metric}): {row[primary_metric]:.4f}",
				"Parameters:",
				f"<pre>{row['params_pretty']}</pre>",
			]
			hover_parts.extend([f"{metric}: {row[metric]:.4f}" for metric in metrics])
			fig.add_trace(
				go.Scatter(
					x=list(metrics),
					y=y_values,
					mode="lines+markers",
					name=row["config_id"],
					visible=(family == families[0]),
					hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
				)
			)
			family_trace_indices.append(len(fig.data) - 1)
		traces_per_family.append(family_trace_indices)

	all_visible = [True] * len(fig.data)
	buttons = [
		{
			"label": "All learner families",
			"method": "update",
			"args": [
				{"visible": all_visible},
				{
					"title": f"Metric profiles — all learner families ({'normalized' if normalize else 'raw'} values)"
				},
			],
		}
	]

	for family, indices in zip(families, traces_per_family):
		visible = [False] * len(fig.data)
		for idx in indices:
			visible[idx] = True
		buttons.append(
			{
				"label": family,
				"method": "update",
				"args": [
					{"visible": visible},
					{
						"title": f"Metric profiles — {family} ({'normalized' if normalize else 'raw'} values)"
					},
				],
			}
		)

	fig.update_layout(
		title=f"Metric profiles — all learner families ({'normalized' if normalize else 'raw'} values)",
		xaxis_title="Metrics",
		yaxis_title="Normalized value" if normalize else "Raw value",
		updatemenus=[
			{
				"buttons": buttons,
				"direction": "down",
				"showactive": True,
				"x": 1.02,
				"y": 1.12,
				"xanchor": "left",
				"yanchor": "top",
			}
		],
		height=650,
		margin=dict(l=70, r=220, t=80, b=70),
		legend_title="Configuration",
	)
	return fig


def build_report_data(df: pd.DataFrame, metrics: Sequence[str]) -> dict:
	records = []
	for _, row in df.iterrows():
		records.append(
			{
				"config_id": row["config_id"],
				"learner": row["learner"],
				"learner_abbr": row["learner_abbr"],
				"learner_order": int(row["learner_order"]),
				"config_index": int(row["config_index"]),
				"params_json": row["params_json"],
				"params_pretty": row["params_pretty"],
				"metrics": {metric: float(row[metric]) for metric in metrics},
				"metrics_norm": {metric: float(row[f"{metric}__norm"]) for metric in metrics},
			}
		)
	return {
		"metrics": list(metrics),
		"default_primary_metric": DEFAULT_PRIMARY_METRIC
		if DEFAULT_PRIMARY_METRIC in metrics
		else metrics[0],
		"default_normalize": DEFAULT_NORMALIZE,
		"records": records,
	}


def save_csv(df: pd.DataFrame, metrics: Sequence[str], output_dir: Path) -> Path:
	cols = ["config_id", "learner", "learner_abbr", "params_json", *metrics]
	path = output_dir / "interactive_summary.csv"
	df[cols].to_csv(path, index=False)
	return path


def make_html_report(df: pd.DataFrame, metrics: Sequence[str], output_dir: Path) -> Path:
	report_data = build_report_data(df, metrics)
	primary_metric = report_data["default_primary_metric"]
	normalize = report_data["default_normalize"]

	heatmap_html = make_heatmap_figure(
		df, metrics, normalize=normalize, primary_metric=primary_metric
	).to_html(
		full_html=False,
		include_plotlyjs=False,
		div_id="heatmap_plot",
	)
	profile_html = make_profile_figure(
		df, metrics, normalize=normalize, primary_metric=primary_metric
	).to_html(
		full_html=False,
		include_plotlyjs=False,
		div_id="profile_plot",
	)

	metrics_options = "\n".join(
		f'<option value="{m}" {"selected" if m == primary_metric else ""}>{m}</option>'
		for m in metrics
	)

	html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive learner multidimensional report</title>
<style>
    :root {
        --border: #d9d9de;
        --muted: #6b7280;
        --bg-soft: #f8fafc;
        --accent: #0f62fe;
    }
    body {
        font-family: Arial, sans-serif;
        margin: 24px;
        line-height: 1.45;
        color: #111827;
    }
    h1, h2, h3 { margin-bottom: 0.35em; }
    .top-grid {
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 18px;
        align-items: start;
    }
    .card {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 18px;
        background: white;
    }
    .soft { background: var(--bg-soft); }
    .controls {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        align-items: end;
    }
    label {
        display: block;
        font-weight: bold;
        margin-bottom: 6px;
    }
    select, input[type="checkbox"] {
        font-size: 14px;
    }
    .inline-help {
        display: inline-block;
        font-weight: bold;
        color: var(--accent);
        margin-left: 8px;
        cursor: default;
    }
    details {
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        background: #fcfcfd;
    }
    details summary {
        cursor: pointer;
        font-weight: bold;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 12px;
    }
    .stat-box {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        background: #fff;
    }
    .muted { color: var(--muted); }
    .mono { font-family: Consolas, Menlo, monospace; white-space: pre-wrap; }
    .small { font-size: 13px; }
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }
    th, td {
        border: 1px solid var(--border);
        padding: 8px;
        vertical-align: top;
        text-align: left;
    }
    th {
        background: #f3f4f6;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .table-wrap {
        max-height: 480px;
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 10px;
    }
    .footer-note {
        color: var(--muted);
        font-size: 13px;
    }
    @media (max-width: 1000px) {
        .top-grid { grid-template-columns: 1fr; }
    }
</style>
</head>
<body>

<h1>Interactive learner configuration report</h1>

<div class="top-grid">
    <section class="card soft">
        <h2>Controls <span class="inline-help" title="Change the primary ranking metric and switch between raw and normalized chart values.">ⓘ</span></h2>
        <div class="controls">
            <div>
                <label for="primaryMetricSelect">Primary metric</label>
                <select id="primaryMetricSelect">__METRIC_OPTIONS__</select>
            </div>
            <div>
                <label for="normalizeToggle">Normalize chart values</label>
                <input id="normalizeToggle" type="checkbox" __NORMALIZE_CHECKED__>
                <span class="small muted">Recommended when metrics have different scales.</span>
            </div>
            <div>
                <label for="searchInput">Find configuration</label>
                <input id="searchInput" type="text" placeholder="e.g. LR2 or random-forest" style="width:100%; padding:8px; box-sizing:border-box;">
            </div>
        </div>
        <p class="footer-note">
            The report keeps learner families grouped together. Inside each family, configurations are sorted by the selected primary metric.
        </p>
    </section>

    <section class="card">
        <h2>How to read the report <span class="inline-help" title="These notes explain what each panel means and why normalization can matter.">ⓘ</span></h2>
        <details open>
            <summary>Interpretation guide</summary>
            <ul>
                <li><strong>Heatmap</strong>: each row is one configuration such as LR1 or RF2, each column is one metric, and darker or brighter color means stronger score according to the selected scale.</li>
                <li><strong>Metric profile chart</strong>: each line is one configuration moving across the metric axes. Similar lines mean similar behavior. A line that stays high across metrics is usually a strong candidate.</li>
                <li><strong>Ranking table</strong>: this is where you see the exact parameter settings, not just LR1 or RF2. Use it as the decoding table for every short config label.</li>
                <li><strong>Primary metric</strong>: this is your decision anchor. The report re-sorts configurations according to it, but all other metrics remain visible so you can inspect trade-offs.</li>
                <li><strong>Normalization</strong>: not strictly required, but strongly recommended for multidimensional visuals when metrics do not share the same scale. In your example, MCC is naturally different from metrics like F1 and AUC, so normalization helps make colors and shapes comparable.</li>
            </ul>
        </details>
    </section>
</div>

<section class="card">
    <h2>Metric notes</h2>
    <ul>
        __METRIC_EXPLANATIONS__
    </ul>
</section>

<section class="card">
    <h2>Current best results</h2>
    <div id="summaryCards" class="stat-grid"></div>
</section>

<section class="card">
    <h2>Heatmap overview <span class="inline-help" title="Hover a cell to see learner name, full parameters, raw value, and normalized value.">ⓘ</span></h2>
    <div id="heatmapContainer">__HEATMAP_HTML__</div>
</section>

<section class="card">
    <h2>Metric profiles <span class="inline-help" title="Use the dropdown built into the chart to view all learner families or focus on a single family.">ⓘ</span></h2>
    <div id="profileContainer">__PROFILE_HTML__</div>
</section>

<section class="card">
    <h2>Ranking table and parameter decoder <span class="inline-help" title="This table is the direct answer to: what exactly is LR1 or RF2? It shows the full parameter settings.">ⓘ</span></h2>
    <div class="table-wrap">
        <table id="rankingTable"></table>
    </div>
</section>

<section class="card">
    <h2>Why LR1 alone is not enough</h2>
    <p>
        Think of <strong>LR1</strong> as a short laboratory code, not as the real experimental condition.
        The real experimental condition is the pair <strong>learner family + parameter settings</strong>.
        So the report keeps the short code for compact plotting, but the hover panels and ranking table always reveal the exact parameters behind that code.
    </p>
</section>

<script>__PLOTLY_JS__</script>
<script>
const REPORT_DATA = __REPORT_JSON__;

function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function orderRecords(records, primaryMetric) {
    return deepCopy(records).sort((a, b) => {
        if (a.learner_order !== b.learner_order) return a.learner_order - b.learner_order;
        const diff = b.metrics[primaryMetric] - a.metrics[primaryMetric];
        if (diff !== 0) return diff;
        return a.config_index - b.config_index;
    });
}

function filteredRecords(records, query) {
    const q = (query || '').trim().toLowerCase();
    if (!q) return records;
    return records.filter(r =>
        r.config_id.toLowerCase().includes(q) ||
        r.learner.toLowerCase().includes(q) ||
        r.params_json.toLowerCase().includes(q)
    );
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatParamsCell(paramsJson) {
    return `<pre class="mono small">${escapeHtml(paramsJson)}</pre>`;
}

function metricListHtml(record) {
    const items = REPORT_DATA.metrics.map(m => `${m}=${record.metrics[m].toFixed(4)}`).join('<br>');
    return `<div class="small">${items}</div>`;
}

function renderSummaryCards(records, primaryMetric) {
    const container = document.getElementById('summaryCards');
    if (!records.length) {
        container.innerHTML = '<div class="stat-box">No rows match the current filter.</div>';
        return;
    }

    const sortedGlobal = deepCopy(records).sort((a, b) => b.metrics[primaryMetric] - a.metrics[primaryMetric]);
    const bestOverall = sortedGlobal[0];

    const byFamily = {};
    for (const rec of records) {
        if (!byFamily[rec.learner] || rec.metrics[primaryMetric] > byFamily[rec.learner].metrics[primaryMetric]) {
            byFamily[rec.learner] = rec;
        }
    }

    let html = `
        <div class="stat-box">
            <h3>Best overall by ${primaryMetric}</h3>
            <div><strong>${bestOverall.config_id}</strong> — ${bestOverall.learner}</div>
            <div>${primaryMetric} = <strong>${bestOverall.metrics[primaryMetric].toFixed(4)}</strong></div>
            <div class="small muted">Parameters</div>
            ${formatParamsCell(bestOverall.params_json)}
        </div>
    `;

    for (const [family, rec] of Object.entries(byFamily)) {
        html += `
            <div class="stat-box">
                <h3>Best in ${family}</h3>
                <div><strong>${rec.config_id}</strong></div>
                <div>${primaryMetric} = <strong>${rec.metrics[primaryMetric].toFixed(4)}</strong></div>
                <div class="small muted">Parameters</div>
                ${formatParamsCell(rec.params_json)}
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderRankingTable(records, primaryMetric) {
    const table = document.getElementById('rankingTable');
    const sorted = deepCopy(records).sort((a, b) => b.metrics[primaryMetric] - a.metrics[primaryMetric]);

    let html = '<thead><tr>' +
        '<th>Global rank</th>' +
        '<th>Config</th>' +
        '<th>Learner</th>' +
        `<th>${primaryMetric}</th>` +
        '<th>All metrics</th>' +
        '<th>Parameters</th>' +
        '</tr></thead><tbody>';

    sorted.forEach((rec, idx) => {
        html += '<tr>' +
            `<td>${idx + 1}</td>` +
            `<td><strong>${rec.config_id}</strong></td>` +
            `<td>${rec.learner}</td>` +
            `<td>${rec.metrics[primaryMetric].toFixed(4)}</td>` +
            `<td>${metricListHtml(rec)}</td>` +
            `<td>${formatParamsCell(rec.params_json)}</td>` +
            '</tr>';
    });

    html += '</tbody>';
    table.innerHTML = html;
}

function makeHeatmapFigure(records, primaryMetric, normalize) {
    const ordered = orderRecords(records, primaryMetric);
    const metrics = REPORT_DATA.metrics;

    const z = [];
    const text = [];
    const customdata = [];
    const y = [];

    for (const rec of ordered) {
        y.push(rec.config_id);
        const zRow = [];
        const textRow = [];
        const customRow = [];
        for (const metric of metrics) {
            const raw = rec.metrics[metric];
            const norm = rec.metrics_norm[metric];
            zRow.push(normalize ? norm : raw);
            textRow.push(raw.toFixed(3));
            customRow.push([rec.config_id, rec.learner, rec.params_pretty, metric, raw, norm]);
        }
        z.push(zRow);
        text.push(textRow);
        customdata.push(customRow);
    }

    return {
        data: [{
            type: 'heatmap',
            x: metrics,
            y: y,
            z: z,
            text: text,
            texttemplate: '%{text}',
            customdata: customdata,
            colorscale: 'Viridis',
            colorbar: { title: normalize ? 'Normalized' : 'Raw' },
            hovertemplate:
                '<b>%{customdata[0]}</b><br>' +
                'Learner: %{customdata[1]}<br>' +
                'Metric: %{customdata[3]}<br>' +
                'Raw value: %{customdata[4]:.4f}<br>' +
                'Normalized: %{customdata[5]:.4f}<br>' +
                '<br><b>Parameters</b><br><pre>%{customdata[2]}</pre>' +
                '<extra></extra>'
        }],
        layout: {
            title: `Heatmap overview — grouped by learner family, sorted within family by ${primaryMetric}`,
            xaxis: { title: 'Metrics' },
            yaxis: { title: 'Configurations' },
            height: Math.max(550, 40 * ordered.length + 220),
            margin: { l: 90, r: 20, t: 80, b: 70 }
        }
    };
}

function makeProfileFigure(records, primaryMetric, normalize) {
    const ordered = orderRecords(records, primaryMetric);
    const metrics = REPORT_DATA.metrics;
    const families = [...new Set(ordered.map(r => r.learner))];
    const data = [];

    for (const family of families) {
        const familyRows = ordered.filter(r => r.learner === family);
        for (const rec of familyRows) {
            const y = metrics.map(m => normalize ? rec.metrics_norm[m] : rec.metrics[m]);
            const hoverLines = [
                `<b>${rec.config_id}</b>`,
                `Learner: ${rec.learner}`,
                `Primary metric (${primaryMetric}): ${rec.metrics[primaryMetric].toFixed(4)}`,
                'Parameters:',
                `<pre>${escapeHtml(rec.params_pretty)}</pre>`
            ];
            metrics.forEach(m => hoverLines.push(`${m}: ${rec.metrics[m].toFixed(4)}`));
            data.push({
                type: 'scatter',
                mode: 'lines+markers',
                x: metrics,
                y: y,
                name: rec.config_id,
                legendgroup: family,
                visible: true,
                customfamily: family,
                hovertemplate: hoverLines.join('<br>') + '<extra></extra>'
            });
        }
    }

    const buttons = [
        {
            label: 'All learner families',
            method: 'update',
            args: [{ visible: data.map(() => true) }, { title: `Metric profiles — all learner families (${normalize ? 'normalized' : 'raw'} values)` }]
        }
    ];

    for (const family of families) {
        const visible = data.map(trace => trace.customfamily === family);
        buttons.push({
            label: family,
            method: 'update',
            args: [{ visible: visible }, { title: `Metric profiles — ${family} (${normalize ? 'normalized' : 'raw'} values)` }]
        });
    }

    return {
        data: data,
        layout: {
            title: `Metric profiles — all learner families (${normalize ? 'normalized' : 'raw'} values)`,
            xaxis: { title: 'Metrics' },
            yaxis: { title: normalize ? 'Normalized value' : 'Raw value' },
            updatemenus: [{
                buttons: buttons,
                direction: 'down',
                showactive: true,
                x: 1.02,
                y: 1.12,
                xanchor: 'left',
                yanchor: 'top'
            }],
            height: 650,
            margin: { l: 70, r: 220, t: 80, b: 70 },
            legend: { title: { text: 'Configuration' } }
        }
    };
}

function updateEverything() {
    const primaryMetric = document.getElementById('primaryMetricSelect').value;
    const normalize = document.getElementById('normalizeToggle').checked;
    const query = document.getElementById('searchInput').value;

    const visibleRecords = filteredRecords(REPORT_DATA.records, query);

    renderSummaryCards(visibleRecords, primaryMetric);
    renderRankingTable(visibleRecords, primaryMetric);

    const heatmap = makeHeatmapFigure(visibleRecords, primaryMetric, normalize);
    Plotly.react('heatmap_plot', heatmap.data, heatmap.layout, {responsive: true});

    const profile = makeProfileFigure(visibleRecords, primaryMetric, normalize);
    Plotly.react('profile_plot', profile.data, profile.layout, {responsive: true});
}

document.getElementById('primaryMetricSelect').addEventListener('change', updateEverything);
document.getElementById('normalizeToggle').addEventListener('change', updateEverything);
document.getElementById('searchInput').addEventListener('input', updateEverything);
updateEverything();
</script>
</body>
</html>
"""

	html = html.replace("__METRIC_OPTIONS__", metrics_options)
	html = html.replace("__NORMALIZE_CHECKED__", "checked" if normalize else "")
	html = html.replace("__METRIC_EXPLANATIONS__", metric_explanation_block(metrics))
	html = html.replace("__HEATMAP_HTML__", heatmap_html)
	html = html.replace("__PROFILE_HTML__", profile_html)
	html = html.replace("__PLOTLY_JS__", get_plotlyjs())
	html = html.replace("__REPORT_JSON__", json.dumps(report_data, ensure_ascii=False))

	path = output_dir / "interactive_report.html"
	path.write_text(html, encoding="utf-8")
	return path


def main(results: Sequence[Dict[str, Any]] | None = None) -> None:
	csv_path = Path(__file__).resolve().parent.parent.parent / "evaluation_results.csv"

	if csv_path.exists():
		df = pd.read_csv(csv_path)
		results = df.to_dict("records")
	else:
		results = results or SAMPLE_RESULTS
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	df, metrics = build_dataframe(results)
	csv_path = save_csv(df, metrics, OUTPUT_DIR)
	html_path = make_html_report(df, metrics, OUTPUT_DIR)

	print("Detected metrics:", ", ".join(metrics))
	print("Wrote:")
	print(" -", csv_path)
	print(" -", html_path)


if __name__ == "__main__":
	main()

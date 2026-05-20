from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from .utils import root

TEMPLATES: Path = root('templates')

def heatmap(df: pd.DataFrame):
    df = df.set_index("Learner")
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=df.values,
                x=df.columns.tolist(),
                y=df.index.tolist(),
                colorscale="Viridis",
                text=[[f"{value:.5f}" for value in row] for row in df.values],
                texttemplate="%{text}",
                # hovertemplate=(
                #     "<b>Learner: %{y}</b><br>"
                #     "%{x}: %{z}<br>"
                #     "<extra></extra>"
                # ),
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


def main(experiment: int):
    # 1) Load the results (CSV file)
    fd = root("results", f"experiment{experiment}-evaluation-results.csv")
    if not fd.exists():
        raise FileNotFoundError(f"Results file not found: {fd}")

    df = pd.read_csv(fd)
    logger.debug(df)

    # 2) Generate heatmap
    hmap = heatmap(df)
    table = df.to_html(index=False, classes="table table-striped table-bordered")
    table_html = f'<div style="overflow-x: auto; width: 100%; margin: 0;">{table}</div>'

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template('report.html')

    html = template.render(
        report_title=f"Experiment {experiment} report",
        report_description="TODO: write some explanations here",
        evaluation_results=table_html,
        heatmap=hmap,
        metrics_notes="TODO: write some explanations here"
    )

    output_file: Path = root('reports', f'experiment{experiment}.html')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding='utf-8')

    logger.success(f"HTML report generated: {output_file}")

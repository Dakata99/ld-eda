"use strict";

document.addEventListener("DOMContentLoaded", function () {
    const table = document.getElementById("results-table");
    if (!table) return;

    const headers = table.querySelectorAll("th");

    headers.forEach((header, columnIndex) => {
        header.style.cursor = "pointer";

        header.addEventListener("click", () => {
            const tbody = table.querySelector("tbody");
            const rows = Array.from(tbody.querySelectorAll("tr"));

            const currentDirection = header.dataset.sortDirection || "asc";
            const newDirection = currentDirection === "asc" ? "desc" : "asc";

            headers.forEach(h => h.dataset.sortDirection = "");
            header.dataset.sortDirection = newDirection;

            rows.sort((rowA, rowB) => {
                const cellA = rowA.children[columnIndex].innerText.trim();
                const cellB = rowB.children[columnIndex].innerText.trim();

                const valueA = parseFloat(cellA);
                const valueB = parseFloat(cellB);

                const bothNumeric = !isNaN(valueA) && !isNaN(valueB);

                if (bothNumeric) {
                    return newDirection === "asc"
                        ? valueA - valueB
                        : valueB - valueA;
                }

                return newDirection === "asc"
                    ? cellA.localeCompare(cellB)
                    : cellB.localeCompare(cellA);
            });

            rows.forEach(row => tbody.appendChild(row));
        });
    });
});

/**
 * Evaluation Results Row Comparison
 *
 * Usage:
 * - Click one row to select the baseline model.
 * - Ctrl/Cmd + click another row to select the candidate model.
 * - Delta is calculated as:
 *
 *      Candidate - Baseline
 *
 * In practice:
 * - Row A = first selected row
 * - Row B = second selected row
 * - Delta = Row B - Row A
 */

document.addEventListener("DOMContentLoaded", function () {
    const wrapper = document.querySelector("#evaluation-results-wrapper");

    if (!wrapper) {
        console.warn("Evaluation results wrapper not found.");
        return;
    }

    const table = wrapper.querySelector("table");

    if (!table) {
        console.warn("No table found inside evaluation results wrapper.");
        return;
    }

    const theadRow = table.querySelector("thead tr");
    const tbody = table.querySelector("tbody");
    const comparisonResult = document.querySelector("#comparison-result");

    if (!theadRow || !tbody || !comparisonResult) {
        console.warn("Table header, body, or comparison result box missing.");
        return;
    }

    table.classList.add(
        "table",
        "table-striped",
        "table-bordered",
        "results-table"
    );

    const selectedRows = [];
    const rows = Array.from(tbody.querySelectorAll("tr"));

    rows.forEach(function (row) {
        row.classList.add("selectable-row");
        row.setAttribute("title", "Click to select. Ctrl/Cmd + click to compare.");
    });

    tbody.addEventListener("click", function (event) {
        const row = event.target.closest("tr");

        if (!row || !tbody.contains(row)) {
            return;
        }

        const isMultiSelect = event.ctrlKey || event.metaKey;

        if (!isMultiSelect) {
            clearSelection(selectedRows);
            selectRow(row, selectedRows);
            updateComparison(table, theadRow, selectedRows, comparisonResult);
            return;
        }

        if (selectedRows.includes(row)) {
            unselectRow(row, selectedRows);
            updateComparison(table, theadRow, selectedRows, comparisonResult);
            return;
        }

        if (selectedRows.length >= 2) {
            alert("Please select only two rows for comparison.");
            return;
        }

        selectRow(row, selectedRows);
        updateComparison(table, theadRow, selectedRows, comparisonResult);
    });

    updateComparison(table, theadRow, selectedRows, comparisonResult);
});

/**
 * Selects a table row.
 */
function selectRow(row, selectedRows) {
    row.classList.add("selected-row");
    selectedRows.push(row);
}

/**
 * Unselects a table row.
 */
function unselectRow(row, selectedRows) {
    row.classList.remove("selected-row");

    const index = selectedRows.indexOf(row);

    if (index !== -1) {
        selectedRows.splice(index, 1);
    }
}

/**
 * Clears all selected rows.
 */
function clearSelection(selectedRows) {
    selectedRows.forEach(function (row) {
        row.classList.remove("selected-row");
    });

    selectedRows.length = 0;
}

/**
 * Parses a numeric value from a table cell.
 *
 * Supports:
 * - 0.812345
 * - 0,812345
 * - values with spaces
 */
function parseNumber(value) {
    if (value === null || value === undefined) {
        return null;
    }

    const cleaned = value
        .trim()
        .replace(",", ".")
        .replace(/\s/g, "");

    if (cleaned === "") {
        return null;
    }

    const number = Number(cleaned);

    return Number.isFinite(number) ? number : null;
}

/**
 * Returns the text content of a table cell by index.
 */
function getCellText(row, index) {
    const cells = Array.from(row.querySelectorAll("td"));

    if (!cells[index]) {
        return "";
    }

    return cells[index].textContent.trim();
}

/**
 * Escapes HTML to avoid injecting raw table content into generated HTML.
 */
function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Formats a number with a sign.
 */
function formatSignedNumber(value, digits) {
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(digits)}`;
}

/**
 * Interprets the absolute size of a metric delta.
 *
 * Important:
 * These thresholds express practical difference,
 * not statistical significance.
 */
function interpretDelta(delta) {
    const absDelta = Math.abs(delta);

    let strength = "";

    if (absDelta < 0.001) {
        strength = "Negligible";
    } else if (absDelta < 0.005) {
        strength = "Very small";
    } else if (absDelta < 0.010) {
        strength = "Small";
    } else if (absDelta < 0.050) {
        strength = "Moderate";
    } else {
        strength = "Large";
    }

    if (delta > 0) {
        return `${strength} improvement`;
    }

    if (delta < 0) {
        return `${strength} decrease`;
    }

    return "No difference";
}

/**
 * Returns a CSS class based on delta direction.
 */
function getDeltaClass(delta) {
    if (delta > 0) {
        return "positive-delta";
    }

    if (delta < 0) {
        return "negative-delta";
    }

    return "neutral-delta";
}

/**
 * Returns a CSS class based on practical impact size.
 */
function getImpactClass(delta) {
    const absDelta = Math.abs(delta);

    if (absDelta < 0.001) {
        return "negligible-impact";
    }

    if (absDelta < 0.010) {
        return "small-impact";
    }

    if (absDelta < 0.050) {
        return "meaningful-impact";
    }

    return "large-impact";
}

/**
 * Attempts to create a useful row label for the selected model.
 *
 * It searches for common identifier columns such as:
 * - Model
 * - Learner
 * - ID
 * - Family
 */
function getRowLabel(row, headers) {
    const preferredColumns = [
        "ID",
        "Id",
        "Model ID",
        "Learner ID",
        "Learner",
        "Model",
        "Family"
    ];

    for (const columnName of preferredColumns) {
        const index = headers.findIndex(function (header) {
            return header.toLowerCase() === columnName.toLowerCase();
        });

        if (index !== -1) {
            const value = getCellText(row, index);

            if (value !== "") {
                return value;
            }
        }
    }

    const firstCell = getCellText(row, 0);

    return firstCell !== "" ? firstCell : "Selected row";
}

/**
 * Updates the comparison box.
 */
function updateComparison(table, theadRow, selectedRows, comparisonResult) {
    if (selectedRows.length !== 2) {
        return;
    }

    const rowA = selectedRows[0];
    const rowB = selectedRows[1];

    const headers = Array.from(theadRow.querySelectorAll("th")).map(function (th) {
        return th.textContent.trim();
    });

    const rowALabel = escapeHtml(getRowLabel(rowA, headers));
    const rowBLabel = escapeHtml(getRowLabel(rowB, headers));

    let html = `
        <table class="delta-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline (${rowALabel})</th>
                    <th>Candidate (${rowBLabel})</th>
                    <th>Delta (${rowBLabel} - ${rowALabel})</th>
                    <th>Delta (%)</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
    `;

    let numericColumnsFound = 0;

    for (let i = 0; i < headers.length; i++) {
        const header = headers[i];

        const valueA = getCellText(rowA, i);
        const valueB = getCellText(rowB, i);

        const numA = parseNumber(valueA);
        const numB = parseNumber(valueB);

        /**
         * Skip non-numeric columns:
         * - Learner
         * - Family
         * - ID
         * - textual model configuration
         */
        if (numA === null || numB === null) {
            continue;
        }

        numericColumnsFound += 1;

        const delta = numB - numA;
        const relativeDelta = numA !== 0 ? (delta / Math.abs(numA)) * 100 : null;

        const deltaClass = getDeltaClass(delta);
        const impactClass = getImpactClass(delta);
        const interpretation = interpretDelta(delta);

        html += `
            <tr>
                <td>${escapeHtml(header)}</td>
                <td>${numA.toFixed(6)}</td>
                <td>${numB.toFixed(6)}</td>
                <td class="${deltaClass}">
                    ${formatSignedNumber(delta, 6)}
                </td>
                <td class="${deltaClass}">
                    ${relativeDelta !== null
                ? `${formatSignedNumber(relativeDelta, 3)}%`
                : "N/A"
            }
                </td>
                <td class="delta-interpretation ${impactClass}">
                    ${escapeHtml(interpretation)}
                </td>
            </tr>
        `;
    }

    if (numericColumnsFound === 0) {
        html += `
            <tr>
                <td colspan="6">
                    No numeric metric columns were found for comparison.
                </td>
            </tr>
        `;
    }

    html += `
            </tbody>
        </table>
    `;

    comparisonResult.innerHTML = html;
}
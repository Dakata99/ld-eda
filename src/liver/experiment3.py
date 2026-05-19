"""
TODO: add docstring.
Binary classification for all 3 dataset.
"""

import json
import numpy as np
import collections
from loguru import logger


from .load import load_csv, load_configuration
from .utils import create_learners

from Orange.preprocess import (
    PreprocessorList,
    Impute,
    Average,
    Continuize,
    Normalize
)

import numpy as np
from Orange.data import Domain, StringVariable, Table, DiscreteVariable

class EditDomain:
    def __init__(self):
        pass

    def remap(self,
              data: Table,
              attribute: str,
              mapping: dict[str, str],
              new_attr_name: str | None = None
              ):
        """
        Remap values of a discrete attribute/feature.

        Example:
            mapping = {
                "1": "Male",
                "2": "Female",
            }
        """
        old_attrs = list(data.domain.attributes)
        old_var = data.domain[attribute]

        if old_var not in old_attrs:
            raise ValueError(
                f"'{attribute}' is not an attribute. "
                "This function remaps feature columns, not class/metas."
            )

        if not old_var.is_discrete:
            raise TypeError(f"'{attribute}' is not discrete.")

        col_index = old_attrs.index(old_var)

        old_values = list(old_var.values)
        logger.debug(f"Old values for '{attribute}': {old_values}")

        new_labels = []
        for label in old_values:
            new_label = mapping.get(label, label)
            if new_label not in new_labels:
                new_labels.append(new_label)

        new_var = DiscreteVariable(
            new_attr_name or old_var.name,
            values=new_labels,
        )

        X_new = data.X.copy()
        old_col = X_new[:, col_index].copy()
        new_col = np.full(len(data), np.nan, dtype=float)

        label_to_new_index = {
            label: i for i, label in enumerate(new_labels)
        }

        for old_index, old_label in enumerate(old_values):
            mapped_label = mapping.get(old_label, old_label)
            mapped_index = label_to_new_index[mapped_label]

            new_col[old_col == old_index] = mapped_index

        X_new[:, col_index] = new_col

        new_attrs = old_attrs.copy()
        new_attrs[col_index] = new_var

        new_domain = Domain(
            attributes=new_attrs,
            class_vars=data.domain.class_var,
            metas=data.domain.metas,
        )

        return Table.from_numpy(
            domain=new_domain,
            X=X_new,
            Y=data.Y.copy(),
            metas=data.metas.copy(),
            W=data.W.copy() if data.has_weights() else None,
            ids=data.ids.copy() if hasattr(data, "ids") else None,
        )

    def rename(self, data: Table, mapping: dict[str, str]):
        """Rename a feature in the dataset."""

        for old_name, new_name in mapping.items():
            old_attrs = list(data.domain.attributes)
            logger.debug(f"Old attributes: {[attr.name for attr in old_attrs]}")
            old_var = data.domain[old_name]

            if old_var not in old_attrs:
                raise ValueError(f"'{old_name}' is not an attribute.")

            col_index = old_attrs.index(old_var)

            new_var = DiscreteVariable(
                new_name,
                values=list(old_var.values),
            )

            X_new = data.X.copy()
            new_attrs = old_attrs.copy()
            new_attrs[col_index] = new_var

        new_domain = Domain(
            attributes=new_attrs,
            class_vars=data.domain.class_var,
            metas=data.domain.metas,
        )

        return Table.from_numpy(
            domain=new_domain,
            X=X_new,
            Y=data.Y.copy(),
            metas=data.metas.copy(),
            W=data.W.copy() if data.has_weights() else None,
            ids=data.ids.copy() if hasattr(data, "ids") else None,
        )

    def rename_target(self, data: Table, new_name: str):
        old_class = data.domain.class_var

        new_class = DiscreteVariable(
            name=new_name,          # new name
            values=old_class.values  # keep same values
        )

        new_domain = Domain(data.domain.attributes, new_class, data.domain.metas)

        # Reuse X, Y and metas directly — no transform needed
        data = Table.from_numpy(
            new_domain,
            data.X,
            data.Y,
            data.metas if data.domain.metas else None
        )

        # Verify
        print(data.domain.class_var)  # Disease
        print(data.Y[:5])             # same values as before

        return data

class Concatenate:
    def __init__(self):
        pass

    def __call__(self, datasets: list[Table], source_names: list[str]):
        """Concatenate multiple datasets into one dataset."""
        common_attrs = set(datasets[0].domain.attributes)

        for data in datasets[1:]:
            common_attrs &= set(data.domain.attributes)

        logger.debug(f"Common attributes ({len(common_attrs)}): {common_attrs}")

        # Keep original order from first table, filtered to common
        common_vars = [
            var for var in datasets[0].domain.attributes
            if var.name in common_attrs
        ]

        # Build new domain with source meta
        source_meta = StringVariable("Source")

        # Get all unique meta variables across datasets
        all_meta_vars = []
        meta_var_names = set()
        for dataset in datasets:
            for meta_var in (dataset.domain.metas or ()):
                if meta_var.name not in meta_var_names:
                    all_meta_vars.append(meta_var)
                    meta_var_names.add(meta_var.name)

        # Build final domain with all original metas + source
        final_metas = tuple(all_meta_vars) + (source_meta,)
        new_domain = Domain(
            common_vars,
            datasets[0].domain.class_var,
            metas=final_metas
        )

        transformed = []
        for table, source in zip(datasets, source_names):
            # Align columns to common domain (without metas)
            aligned_domain = Domain(common_vars, table.domain.class_var)
            aligned = table.transform(aligned_domain)

            # Build metas: preserve original metas + add source
            metas_list = []

            # Add original metas, padding with None for missing ones
            for meta_var in all_meta_vars:
                if meta_var in table.domain.metas:
                    col_idx = list(table.domain.metas).index(meta_var)
                    metas_list.append(aligned.metas[:, col_idx:col_idx+1] if aligned.metas.shape[1] > 0 else np.full((len(aligned), 1), None, dtype=object))
                else:
                    metas_list.append(np.full((len(aligned), 1), None, dtype=object))

            # Add source column
            source_col = np.array([[source]] * len(aligned), dtype=object)
            metas_list.append(source_col)

            # Concatenate all meta columns
            new_metas = np.hstack(metas_list) if metas_list else None

            transformed.append(
                Table.from_numpy(new_domain, aligned.X, aligned.Y, new_metas)
            )

        # --- 4. Concatenate all ---
        result = Table.concatenate(transformed, axis=0)
        print(f"Result shape: {len(result)} rows x {len(result.domain.attributes)} features")
        return result

def transform(data, target: str):
    """TODO: add docstring."""
    target = data.domain[target]

    assert isinstance(target, DiscreteVariable), "Target variable must be discrete!"
    logger.debug(data.domain)

    features = [
        attr for attr in data.domain.attributes
        if attr.name != target.name
    ]

    domain = Domain(
        attributes=features,
        class_vars=target,
        metas=data.domain.metas
    )

    # Transform to new domain
    data = data.transform(domain)

    info = {
        "target": str(data.domain.class_var),
        "labels": list(data.domain.class_var.values),
        "rows": len(data),
        "features": len(data.domain.attributes),
        "attributes": [str(a) for a in data.domain.attributes],
    }

    logger.debug("Dataset info:\n{}", json.dumps(info, indent=2))
    logger.debug(f"Class variable: {data.domain.class_var}")
    logger.debug(f"Class variable type: {type(data.domain.class_var)}")
    logger.debug(f"Is discrete: {data.domain.class_var.is_discrete}")
    logger.debug(f"Is continuous: {data.domain.class_var.is_continuous}")
    logger.debug(f"Y dtype: {data.Y.dtype}")
    logger.debug(f"Unique Y values: {np.unique(data.Y)[:20]}")

    return data


def main():
    """TODO: write docstring."""
    logger.info('Running experiment 3: binary classification for all 3 datasets')

    # 1) Load the dataset (CSV file)
    indian = load_csv('indian')
    hcv = load_csv('hcv')
    liver = load_csv('liver')

    indian = transform(indian, target="Liver_Disease_Type")
    hcv = transform(hcv, target="Category")
    liver = transform(liver, target="Result")

    logger.warning(f"Indian dataset shape: {indian.domain.class_var}, {len(indian)} rows, {len(indian.domain.attributes)} features")
    logger.warning(f"HCV dataset shape: {hcv.domain.class_var}, {len(hcv)} rows, {len(hcv.domain.attributes)} features")
    logger.warning(f"Liver dataset shape: {liver.domain.class_var}, {len(liver)} rows, {len(liver.domain.attributes)} features")

    ed = EditDomain()
    indian = ed.rename_target(indian, new_name="Condition")
    hcv = ed.rename_target(hcv, new_name="Condition")
    liver = ed.rename_target(liver, new_name="Condition")

    logger.warning(f"Indian dataset shape: {indian.domain.class_var}, {len(indian)} rows, {len(indian.domain.attributes)} features")
    logger.warning(f"HCV dataset shape: {hcv.domain.class_var}, {len(hcv)} rows, {len(hcv.domain.attributes)} features")
    logger.warning(f"Liver dataset shape: {liver.domain.class_var}, {len(liver)} rows, {len(liver.domain.attributes)} features")

    indian.save("indian-before.csv")
    hcv.save("hcv-before.csv")
    liver.save("liver-before.csv")

    # indian = ed.rename
    # liver = ed.rename(liver, mapping={"Age of the patient": "Age"})
    liver = ed.rename(liver, mapping={"Gender of the patient": "Gender"})
    hcv = ed.rename(hcv, mapping={"Sex": "Gender"})
    hcv = ed.remap(hcv, attribute="Gender", mapping={"m": "Male", "f": "Female"})

    indian.save("indian-after.csv")
    hcv.save("hcv-after.csv")
    liver.save("liver-after.csv")

    # 2) Load configuration and create learners
    # config = load_configuration()
    # learners = create_learners(config)
    # logger.debug(learners)

    # 3) Do transformation
    # TODO: INSTEAD OF DOING THIS, JUST EXPORT THE DATA FROM ORANGE WITH THE MAPPED VALUES.
    # THIS IS A LOT OF WORK AND CAN BE ERROR-PRONE.
    # ALSO, THIS IS NOT REALLY THE FOCUS OF THE EXPERIMENT, SO IT'S BETTER TO JUST PREPARE THE DATA IN ORANGE AND THEN LOAD IT HERE.
    # 3.1) Edit Domain of datasets

    # 3.2) Concatenate datasets
    concat = Concatenate()
    data = concat(
        [indian, hcv, liver],
        source_names=["Indian", "HCV", "Liver"]
    )
    logger.debug(f"Concatenated data shape: {data.X.shape}")

    # Check source meta
    sources = [str(row["Source"]) for row in data]
    logger.debug(collections.Counter(sources))
    # e.g. {'dataset_a': 68000, 'dataset_b': 52000, 'dataset_c': 47000}

    # Save
    data.save("merged.csv")

    # 4) Split

    # 5) Preprocess
    preprocessor = PreprocessorList(
        preprocessors=(
            # Average/Most frequent
            Impute(method=Average()),
            # One-hot encoding/One feature per value
            Continuize(multinomial_treatment=Continuize.Indicators),
            # Standardization (z-score normalization)
            Normalize(norm_type=Normalize.NormalizeBySD)
        )
    )

    # TODO: somehow verify that the preprocessor is working correctly (e.g. check for NaNs, check that features are continuous, etc.)
    # data = preprocessor(data)

    # 6) Evaluate

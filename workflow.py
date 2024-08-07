# Import libraries and modules
import html
import io
from textwrap import dedent

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import base64s
import seaborn as sns
from flytekit import (Deck, ImageSpec, Resources, Secret, current_context,
                      task, workflow)
from huggingface_hub import HfApi
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Define the container image to use for the tasks on Union with ImageSpec
image = ImageSpec(
    packages=[
        "scikit-learn==1.4.1.post1",
        "matplotlib==3.8.3",
        "union==0.1.48",
        "seaborn==0.13.2",
        "joblib==1.3.2",
        "huggingface_hub==0.24.0"
    ],
)

# Helper function to convert a matplotlib figure into an HTML string
def _convert_fig_into_html(fig: mpl.figure.Figure) -> str:
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'


# Task: Download the dataset
@task(
    cache=True,
    cache_version="6",
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_dataset() -> pd.DataFrame:
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    return iris_df

# Task: Process the dataset
@task(
    enable_deck=True,
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
)
def process_dataset(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Perform the train-test split
    train_df, test_df = train_test_split(data_df,
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=data_df['target'])

    # Seaborn pairplot full dataset
    pairplot = sns.pairplot(data_df, hue="target")

    metrics_deck = Deck("Metrics")
    metrics_deck.append(_convert_fig_into_html(pairplot))

    return train_df, test_df


# Task: Train a model
@task(
    container_image=image,
    requests=Resources(cpu="3", mem="3Gi"),
)
def train_model(dataset: pd.DataFrame) -> KNeighborsClassifier:
    X_train, y_train = dataset.drop("target", axis="columns"), dataset["target"]
    model = knn = KNeighborsClassifier(n_neighbors=3)
    return model.fit(X_train, y_train)

# Evaluate the model using the test dataset
@task(
    container_image=image,
    enable_deck=True,
    requests=Resources(cpu="2", mem="2Gi"),
)
def evaluate_model(model: KNeighborsClassifier, dataset: pd.DataFrame) -> KNeighborsClassifier:
    ctx = current_context()

    X_test, y_test = dataset.drop("target", axis="columns"), dataset["target"]
    y_pred = model.predict(X_test)

    # Plot confusion matrix in deck
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)

    metrics_deck = Deck("Metrics")
    metrics_deck.append(_convert_fig_into_html(fig))

    # Add classification report
    report = html.escape(classification_report(y_test, y_pred))
    html_report = dedent(
        f"""\
    <h2>Classification report</h2>
    <pre>{report}</pre>"""
    )
    metrics_deck.append(html_report)

    ctx.decks.insert(0, metrics_deck)

    return model

# Upload the model to Hugging Face Hub
@task(
    container_image=image,
    requests=Resources(cpu="1", mem="1Gi"),
    secret_requests=[Secret(group=None, key="hf_token")],
)
def upload_model_to_hf(model: KNeighborsClassifier, repo_name: str, model_name: str) -> str:
    ctx = current_context()
    hf_token = ctx.secrets.get(key="hf_token")
    # Create a new repository (if it doesn't exist)
    api = HfApi()
    api.create_repo(repo_name, token=hf_token, exist_ok=True)

    # save model
    joblib.dump(model, model_name)

    # Upload the model to the HF repository
    api.upload_file(
        path_or_fileobj=model_name,
        path_in_repo=model_name,
        repo_id=repo_name,
        commit_message="Upload model",
        repo_type=None,
        token=hf_token
    )
    return f"Model uploaded to Hugging Face Hub: {repo_name}{model_name}"

# Main workflow that orchestrates the tasks
@workflow
def main(repo_name: str, model_name: str) -> str:
    data_df = download_dataset()
    train, test = process_dataset(data_df)
    model = train_model(dataset=train)
    evaluated_model = evaluate_model(model=model, dataset=test)
    model_name = model_name
    upload_result = upload_model_to_hf(model=evaluated_model,
                                       repo_name=repo_name,
                                       model_name=model_name)
    return upload_result

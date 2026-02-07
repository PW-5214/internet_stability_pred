import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


@st.cache_data
def load_data(path="Internet Speed.csv"):
    df = pd.read_csv(path)
    max_speed = df['Internet_speed'].max()
    bins = [0, 500, 1500, 2500, max_speed + 1]
    labels = ["Very Unstable", "Unstable", "Stable", "Very Stable"]
    df['Stability_Category'] = pd.cut(df['Internet_speed'], bins=bins, labels=labels)
    df['Stability_Category'].replace(["Very Unstable", "Unstable", "Stable", "Very Stable"], [1, 2, 3, 4], inplace=True)
    return df


@st.cache_resource
def train_models(df):
    essential_features = ["Ping_latency", "Download_speed", "Upload_speed", "Packet_loss_rate", "Signal_strength"]
    x = df[essential_features]
    y = df["Stability_Category"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gaussian NB": GaussianNB(),
        "Multinomial NB": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Linear SVM": SVC(kernel='rbf'),
        "Non-Linear SVM": SVC(kernel='poly')
    }

    trained = {}
    accuracies = {}
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred) * 100
            trained[name] = model
            accuracies[name] = acc
        except Exception:
            # skip models that fail to train on this data (e.g., MultinomialNB might require non-negative integers)
            continue

    return trained, accuracies, x_test, y_test


def main():
    st.set_page_config(page_title="Internet Stability Predictor", layout="centered")
    st.title("Internet Stability Predictor")
    st.markdown("Enter your connection metrics and choose a model to predict stability.")

    df = load_data()
    trained_models, model_accuracies, x_test, y_test = train_models(df)

    st.sidebar.header("Input Parameters")
    ping = st.sidebar.number_input("Ping (ms)", min_value=0.0, value=20.0)
    download = st.sidebar.number_input("Download Speed (Mbps)", min_value=0.0, value=100.0)
    upload = st.sidebar.number_input("Upload Speed (Mbps)", min_value=0.0, value=50.0)
    packet_loss = st.sidebar.number_input("Packet Loss (%)", min_value=0.0, value=0.5)
    signal = st.sidebar.number_input("Signal Strength (%)", min_value=0.0, max_value=100.0, value=80.0)

    st.sidebar.header("Model")
    model_choice = st.sidebar.selectbox("Choose model", options=list(trained_models.keys()))

    stability_map = {1: "Very Unstable", 2: "Unstable", 3: "Stable", 4: "Very Stable"}

    if st.button("Predict Stability"):
        features = np.array([ping, download, upload, packet_loss, signal]).reshape(1, -1)
        model = trained_models.get(model_choice)
        if model is None:
            st.error("Selected model is not available (training failed). Choose another model.")
        else:
            pred = model.predict(features)[0]
            st.success(f"Predicted Stability: {stability_map.get(pred, str(pred))}")
            st.info(f"Model accuracy: {model_accuracies.get(model_choice, 0):.2f}%")

    st.header("Model accuracies")
    acc_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
    acc_df = acc_df.sort_values('Accuracy', ascending=False)
    st.table(acc_df)

    st.header("Prediction Distribution & Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        sel = st.selectbox("Choose model for distribution/CM", options=list(trained_models.keys()), key='dist_model')
        if st.button("Show Prediction Distribution"):
            model = trained_models.get(sel)
            if model is not None:
                y_pred = model.predict(x_test)
                classes, counts = np.unique(y_pred, return_counts=True)
                fig, ax = plt.subplots()
                sns.barplot(x=classes, y=counts, palette=['blue', 'green', 'orange', 'red'], ax=ax)
                ax.set_xlabel("Stability Class (1=Very Unstable, 4=Very Stable)")
                ax.set_ylabel("Count")
                ax.set_title(f"Prediction Distribution - {sel}")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Model unavailable")

    with col2:
        sel2 = st.selectbox("Choose model for confusion matrix", options=list(trained_models.keys()), key='cm_model')
        if st.button("Show Confusion Matrix"):
            model = trained_models.get(sel2)
            if model is not None:
                y_pred = model.predict(x_test)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=["Very Unstable","Unstable","Stable","Very Stable"],
                            yticklabels=["Very Unstable","Unstable","Stable","Very Stable"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix - {sel2}")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Model unavailable")

    st.header("Dataset sample")
    st.dataframe(df.head())

    st.markdown("---")
    st.caption("This app was converted from a Tkinter notebook UI to Streamlit for easier web deployment.")


if __name__ == '__main__':
    main()

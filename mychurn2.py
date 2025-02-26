import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

 # Set Streamlit Page Config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Login", "Upload Data", "Data Overview", "Train Model", "Prediction", "Clustering", "Performance Analysis"])

# Authentication State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- LOGIN PAGE ---
if not st.session_state["authenticated"]:
    st.title("🔐 Login Page")

    # User Input for Credentials
    username = st.text_input("Enter your email:", placeholder="temitayoabiola37@gmail.com")
    password = st.text_input("Enter your password:", type="password")

    # Hardcoded User Credentials (For Demo)
    USER_CREDENTIALS = {"temitayoabiola37@gmail.com": "Abiola@37"}

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success("Login successful! Navigate using the sidebar.")
        else:
            st.error("Invalid email or password.")

    st.warning("Please log in to access the features.")
    st.stop()  # Stops further execution if not logged in


st.title("Customer Churn Dashboard")


# Session State for Uploaded File and DataFrame
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "df" not in st.session_state:
    st.session_state.df = None

# File Upload Section
if page == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Store uploaded file and reset file pointer
        st.session_state.uploaded_file = uploaded_file
        uploaded_file.seek(0)

        try:
            df = pd.read_csv(uploaded_file).dropna(how="all")  # Remove empty rows
            st.session_state.df = df  # Store DataFrame in session state

            # Convert data types to ensure compatibility
            df = df.convert_dtypes()  # Automatically infer optimal data types
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")  # Ensure numeric consistency
            non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns
            df[non_numeric_cols] = df[non_numeric_cols].astype(str)  # Ensure string consistency


            st.success("✅ File uploaded successfully!")
            st.write("### Dataset Preview")
            st.write(df.head())

        except pd.errors.EmptyDataError:
            st.error("⚠️ The uploaded file is empty or invalid. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

# Data Overview Section
if page == "Data Overview" and st.session_state.df is not None:
    df = st.session_state.df  # Retrieve stored DataFrame
    st.title("Data Overview & Preprocessing")

    # Dataset Info
    st.write("### Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    st.write("### Null Values Per Column")
    st.write(df.isnull().sum())
    st.write("### Column Data Types")
    st.write(df.dtypes)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    if numeric_cols.empty:
        st.error("⚠️ No numeric columns available for correlation heatmap.")
    else:
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"⚠️ An error occurred while generating the heatmap: {e}")

    # Encode Categorical Columns
    st.write("### Preprocessing: Encoding Categorical Columns")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols:
        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]
        st.success("✅ Categorical columns encoded successfully!")
        st.write("### Processed Dataset Preview")
        st.write(df.head())
    else:
        st.info("No categorical columns detected.")

    # Check for 'Churn' Column
    st.write("### Check for 'Churn' Column")
    if "Churn" not in df.columns:
        st.error("⚠️ No 'Churn' column found! Please check your dataset.")
    else:
        st.success("✅ 'Churn' column found!")

        # Interactive Plotly Visualization
        st.write("### Interactive Plot: Feature vs Churn")
        feature_choice = st.selectbox("Select a Feature", df.columns)
        fig = px.histogram(df, x=feature_choice, color="Churn", barmode="group", title=f"{feature_choice} vs Churn")
        st.plotly_chart(fig)

else:
    st.title(f"{page} Page")
    st.info("This page is under construction or waiting for dataset upload.")

if page == "Train Model":
    st.title("Train Machine Learning Model")

    if "uploaded_file" in st.session_state and st.session_state.uploaded_file and "df" in st.session_state:
        df = st.session_state.df

        if "Churn" not in df.columns:
            st.error("⚠️ No 'Churn' column found! Please upload a dataset with a target variable.")
        else:
            # Preprocess Data
            X = df.drop(columns=["Churn"])
            y = df["Churn"]

            # Handle 'Cluster' column if it exists
            if "Cluster" in X.columns:
                X = X.drop(columns=["Cluster"])

            # Convert all columns to numeric (handle categorical and mixed types)
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Selection
            model_choice = st.selectbox(
                "Choose Model to Train",
                [
                    "Logistic Regression",
                    "Random Forest",
                    "Gradient Boosting",
                    "XGBoost",
                    "Stacking Classifier",
                    "K-Nearest Neighbors (KNN)",
                    "Decision Tree (DT)",
                    "Support Vector Machine (SVM)",
                    "Gaussian Naive Bayes (GNB)"
                ]
            )

            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.svm import SVC
                    from sklearn.naive_bayes import GaussianNB
                    import joblib
                    import os

                    # Create models directory
                    os.makedirs("models", exist_ok=True)

                    # Initialize Model
                    if model_choice == "Logistic Regression":
                        model = LogisticRegression()
                    elif model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
                    elif model_choice == "XGBoost":
                        model = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric="logloss")
                    elif model_choice == "Stacking Classifier":
                        base_models = [
                            ("rf", RandomForestClassifier(n_estimators=20, random_state=42)),
                            ("gb", GradientBoostingClassifier(n_estimators=20, random_state=42))
                        ]
                        model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
                    elif model_choice == "K-Nearest Neighbors (KNN)":
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("knn", KNeighborsClassifier(n_neighbors=5))
                        ])
                    elif model_choice == "Decision Tree (DT)":
                        model = DecisionTreeClassifier(random_state=42)
                    elif model_choice == "Support Vector Machine (SVM)":
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("svm", SVC(probability=True, random_state=42))
                        ])
                    elif model_choice == "Gaussian Naive Bayes (GNB)":
                        model = GaussianNB()

                    # Train Model
                    model.fit(X_train, y_train)

                    # Save Model
                    model_path = f"models/{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
                    joblib.dump(model, model_path)

                    st.success(f"✅ Training Complete! Model saved as: {model_path}")
    else:
        st.error("⚠️ No dataset uploaded! Please go to the 'Upload Data' page to upload a dataset.")

            # 4️⃣ Prediction Page (Left Column for Input, Right Column for Feature Importance)
if page == "Prediction":
    st.title("Customer Churn Prediction")

    # Check if the dataset is uploaded and processed
    if "uploaded_file" in st.session_state and st.session_state.uploaded_file and "df" in st.session_state:
        df = st.session_state.df  # Retrieve the DataFrame

        col1, col2 = st.columns(2)

        # Left Column: User Inputs for Prediction
        with col1:
            st.subheader("Enter Features for Prediction")

            # Prepare Feature Set
            X = df.drop(columns=["Churn"], errors="ignore")  # Drop 'Churn' column if it exists
            feature_names = X.columns  # Keep track of feature names

            # Select Model
            selected_model = st.selectbox(
                "Choose a Trained Model:",
                ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "Stacking Classifier"]
            )
            model_path = f"models/{selected_model.replace(' ', '_')}.pkl"

            try:
                # Attempt to load the trained model
                import joblib
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded: {selected_model}")

                # Input Features
                user_inputs = {}
                for feature in feature_names:
                    user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

                if st.button("Predict"):
                    import numpy as np
                    X_input = np.array([list(user_inputs.values())]).reshape(1, -1)

                    # Ensure the input matches the training feature order
                    X_input = pd.DataFrame(X_input, columns=feature_names)

                    # Prediction
                    prediction = model.predict(X_input)
                    churn_prob = model.predict_proba(X_input)[:, 1][0] * 100

                    if prediction[0] == 1:
                        st.error(f"Churned! (Probability: {churn_prob:.2f}%)")
                    else:
                        st.success(f"Not Churned! (Probability: {churn_prob:.2f}%)")

            except FileNotFoundError:
                st.error("⚠️ The selected model is not available. Please train the model first.")
                model = None  # Ensure `model` is undefined if it fails
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {e}")
                model = None  # Ensure `model` is undefined if it fails

        # Right Column: Feature Importance
        with col2:
            st.subheader("Feature Importance (SHAP)")

            if model is not None:  # Ensure the model is successfully loaded
                try:
                    import shap
                    import matplotlib.pyplot as plt

                    # Select SHAP Explainer Based on Model Type
                    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
                        explainer = shap.TreeExplainer(model)
                    elif isinstance(model, LogisticRegression):
                        explainer = shap.LinearExplainer(model, X)
                    else:
                        explainer = shap.KernelExplainer(model.predict, X)

                    # SHAP Feature Importance
                    shap_values = explainer.shap_values(X)

                    # SHAP Feature Importance Bar Chart
                    st.write("### SHAP Feature Importance")
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"⚠️ Unable to calculate feature importance: {e}")
            else:
                st.error("⚠️ Feature importance cannot be calculated as the model is not available.")
    else:
        st.error("⚠️ No dataset uploaded! Please go to the 'Upload Data' page to upload a dataset.")

import os


# 1️⃣ Performance Analysis Page
if page == "Performance Analysis":
    st.title("Model Performance Metrics")

    # Check if the file has been uploaded and processed
    if "uploaded_file" in st.session_state and st.session_state.uploaded_file and "df" in st.session_state:
        df = st.session_state.df  # Retrieve the DataFrame

        if "Churn" not in df.columns:
            st.error("⚠️ No 'Churn' column found! Please upload a dataset with a target variable.")
        else:
            # Prepare Data
            X = df.drop(columns=["Churn"])
            y = df["Churn"]

            # Exclude the 'Cluster' column if it exists
            if "Cluster" in X.columns:
                X = X.drop(columns=["Cluster"])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Select Model
            model_choice = st.selectbox(
                "Select Model for Analysis",
                [
                    "Logistic Regression", "Random Forest", "Gradient Boosting",
                    "XGBoost", "KNN", "Decision Tree", "SVM",
                    "Gaussian Naive Bayes", "Stacking Classifier"
                ]
            )

            # Load the model based on selection
            model_path = f"models/{model_choice.replace(' ', '_')}.pkl"

            try:
                # Load the model
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded: {model_choice}")

                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # Confusion Matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt)

                # ROC Curve
                st.write("### ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}", color="blue")
                plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random Guess")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.legend(loc="lower right")
                st.pyplot(plt)

                # Feature Importance for Tree-Based Models
                st.write("### Feature Importance (Tree-Based Models)")
                if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
                    try:
                        # Calculate Feature Importance
                        feature_importance = pd.DataFrame({
                            "Feature": X_train.columns,
                            "Importance": model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        # Plot Feature Importance
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
                        plt.title("Feature Importance")
                        plt.xlabel("Importance")
                        plt.ylabel("Feature")
                        st.pyplot(plt)
                    except AttributeError:
                        st.warning("⚠️ Feature importance is not available for the selected model.")

                # Performance Metrics
                st.write("### Model Performance Metrics")
                st.write(f"🔹 **Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
                st.write(f"🔹 **Precision:** {precision_score(y_test, y_pred):.2f}")
                st.write(f"🔹 **Recall:** {recall_score(y_test, y_pred):.2f}")
                st.write(f"🔹 **F1 Score:** {f1_score(y_test, y_pred):.2f}")

            except FileNotFoundError:
                st.error(f"⚠️ Model not found. Please train the '{model_choice}' model first.")
            except ValueError as e:
                st.error(f"⚠️ Feature mismatch error: {e}")
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {e}")
    else:
        st.error("⚠️ No dataset uploaded! Please go to the 'Upload Data' page to upload a dataset.")

# 2️⃣ Clustering Page
if page == "Clustering":
    st.title("Customer Segmentation with Clustering")

    # Check if the dataset is uploaded and processed
    if "uploaded_file" in st.session_state and st.session_state.uploaded_file and "df" in st.session_state:
        df = st.session_state.df  # Retrieve the DataFrame

        # Select Columns for Clustering
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) == 0:
            st.error("⚠️ No numeric columns available for clustering.")
        else:
            st.write("### Numeric Columns for Clustering")
            st.write(numeric_cols)

            selected_features = st.multiselect(
                "Select Features for Clustering:",
                numeric_cols,
                default=list(numeric_cols)[:2]
            )

            if len(selected_features) < 2:
                st.warning("Please select at least two features for clustering.")
            else:
                # User Input: Number of Clusters
                num_clusters = st.slider("Select Number of Clusters (k):", min_value=2, max_value=10, value=3)

                if st.button("Perform Clustering"):
                    with st.spinner("Clustering in progress..."):
                        from sklearn.cluster import KMeans
                        import plotly.express as px

                        # Perform K-Means Clustering
                        X = df[selected_features]
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                        df["Cluster"] = kmeans.fit_predict(X)

                        # Cluster Visualization
                        st.success(f"Clustering completed with {num_clusters} clusters!")

                        # Visualize Clusters in 2D
                        if len(selected_features) >= 2:
                            st.write("### Cluster Visualization")
                            fig = px.scatter(
                                df,
                                x=selected_features[0],
                                y=selected_features[1],
                                color="Cluster",
                                title=f"Clusters Visualization ({selected_features[0]} vs {selected_features[1]})",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig)

                        # Display Cluster Centers
                        st.write("### Cluster Centers")
                        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
                        st.write(cluster_centers)

                        # Show Cluster Assignments
                        st.write("### Dataset with Cluster Assignments")
                        st.write(df.head())
    else:
        st.error("⚠️ No dataset uploaded! Please go to the 'Upload Data' page to upload a dataset.")

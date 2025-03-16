import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Streamlit UI
st.title("CSV File Processing")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

global numeric_col, categorical_col

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")

    # Print Head and Shape
    st.write("Dataset Head:")
    st.write(df.head())
    st.write("Dataset Shape:", df.shape)

    # Identify Numeric & Categorical Columns
    numeric_col = df.select_dtypes(include=['number']).columns.tolist()
    categorical_col = df.select_dtypes(include=['object']).columns.tolist()

    st.write("Numeric Columns:", numeric_col)
    st.write("Categorical Columns:", categorical_col)

    # Check for Missing Values
    missing_values = df.isnull().sum()
    st.write("Missing Values Before Handling:")
    st.write(missing_values)

    # Handle Missing Values
    for col in numeric_col:
        missing_percentage = df[col].isnull().mean() * 100
        if missing_percentage > 40:  # Drop if too many missing values
            df.drop(columns=[col], inplace=True)
        elif missing_percentage > 0:
            if df[col].skew() > 1 or df[col].skew() < -1:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)

    for col in categorical_col:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Verify Missing Values Removed
    missing_values_after = df.isnull().sum()
    st.write("###  Missing Values After Handling:")
    st.write(missing_values_after[missing_values_after > 0])

    st.write("###  Print the Outlier using box plot:")
    for col in numeric_col:
       fig, ax = plt.subplots()  # Create a new figure and axis
       sns.boxplot(x=df[col], ax=ax)  # Use ax to plot
       ax.set_title(f'Box plot of {col}')

       st.pyplot(fig)  # Display the figure in Streamlit

    st.write("### Print the Outlier using Histogram  plot:")

    for col in numeric_col:
        # Calculate quartiles and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define upper and lower bounds for outliers
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        # Filter data for outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        # Create a new figure for Streamlit
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[col], bins=20, color='skyblue', edgecolor='black')

        # Highlight outliers with red markers
        if len(outliers) > 0:
            ax.scatter(outliers, np.zeros_like(outliers) + 0.1, color='red', marker='x', label='Outliers')
            ax.legend()

        ax.set_title(f'Histogram of {col} with Outliers Highlighted')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

        # Display in Streamlit
        st.pyplot(fig)

    target_col = ""
    if (True):
        st.write("### Select Target Column")
        target_col = st.selectbox("Choose the target column:", df.columns)

        if target_col in numeric_col:
            numeric_col.remove(target_col)
        else:
            categorical_col.remove(target_col)

    df_no_outliers = df.copy()
    if(True):


        # Assuming 'df' is your DataFrame and 'numerical_features' is a list of numerical columns

        # Create a copy of the DataFrame to avoid modifying the original

        for col in numeric_col:
            # Calculate quartiles and IQR
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define upper and lower bounds for outliers
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR

            # Filter out outliers
            df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]



        st.write("### Boxplot for no-outlier ")

        for col in numeric_col:
            fig, ax = plt.subplots()  # Create a new figure and axis
            sns.boxplot(x=df_no_outliers[col], ax=ax)  # Use ax to plot
            ax.set_title(f'Box plot of {col}')

            st.pyplot(fig)  # Display the figure in Streamlit

        st.write("### Histogram  for no-outlier ")
        for col in numeric_col:
            # Calculate quartiles and IQR
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define upper and lower bounds for outliers
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR

            # Filter data for outliers
            outliers = df_no_outliers[(df_no_outliers[col] < lower_bound) | (df_no_outliers[col] > upper_bound)][col]

            # Create a new figure for Streamlit
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df_no_outliers[col], bins=20, color='skyblue', edgecolor='black')

            # Highlight outliers with red markers
            if len(outliers) > 0:
                ax.scatter(outliers, np.zeros_like(outliers) + 0.1, color='red', marker='x', label='Outliers')
                ax.legend()

            ax.set_title(f'Histogram of {col} with Outliers Highlighted')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')

            # Display in Streamlit
            st.pyplot(fig)


        st.write('### Filtered data ')
        st.write(df_no_outliers.head())
        st.write(f'### Shape of this dataset:         {df_no_outliers.shape}')

        # Check if target column is categorical or numeric
        if df_no_outliers[target_col].dtype == 'object':
            df_no_outliers[target_col] = LabelEncoder().fit_transform(df_no_outliers[target_col])  # Encode categorical target

        selected_cat_columns = []
        for col in categorical_col:
            if len(df_no_outliers[col].value_counts()) < 10:
                selected_cat_columns.append(col)

        if len(selected_cat_columns) >0 :
          st.title("Categorical Column Visualization")

        # Loop through selected categorical columns and plot count plots
          for col in selected_cat_columns:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot count plot
            sns.countplot(x=df_no_outliers[col], palette='Set2', ax=ax)

            # Annotate percentage on each bar
            total = len(df_no_outliers[col])
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=12, color='black', weight='bold')

            ax.set_title(f'Count Plot of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

            # Show plot in Streamlit
            st.pyplot(fig)

        # Loop through selected categorical columns and plot pie charts
            for col in selected_cat_columns:
              fig, ax = plt.subplots(figsize=(8, 5))

            # Plot pie chart
              df_no_outliers[col].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set2"), ax=ax)

              ax.set_title(f'Pie Chart of {col}')
              ax.set_ylabel("")

            # Show plot in Streamlit
              st.pyplot(fig)

            # Apply Label Encoding to categorical columns
            le = LabelEncoder()
            for col in selected_cat_columns:
                df_no_outliers[col] = le.fit_transform(df_no_outliers[col])

            st.write("### using Label encoder to transform in to numeric ")
            st.write(df_no_outliers.head())


        st.write("### Dealing with Multicollinearity")
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.linear_model import LogisticRegression  # You can change the model here
        from sklearn.feature_selection import RFE






        # Function to calculate VIF
        def calculate_vif(X):
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            return vif_data


        # Check VIF before applying RFE
        st.write("### VIF Before RFE:")
        vif_before = calculate_vif(df_no_outliers[numeric_col])
        st.write(vif_before)

        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import RFE

        # Define the model for RFE
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=5)  # Select top 5 features (adjust as needed)
        X_numeric = df_no_outliers[numeric_col]  # Select only numeric features

        # Fit RFE
        rfe.fit(X_numeric, df_no_outliers[target_col])

        # Get selected features
        selected_features = X_numeric.columns[rfe.support_].tolist()

        st.write("### Selected Features After RFE:")
        st.write(selected_features)

        # Calculate VIF after RFE
        st.write("### VIF After RFE:")
        vif_after = calculate_vif(df_no_outliers[selected_features])
        st.write(vif_after)

        # Create a new DataFrame using selected categorical columns and selected numeric features
        selected_df = df_no_outliers[selected_cat_columns + selected_features]

        # Add the target column
        selected_df[target_col] = df_no_outliers[target_col]

        # Display the final DataFrame
        st.write("### Final Processed DataFrame:")
        st.write(selected_df.head())
        st.write(f"### Shape of the final dataset: {selected_df.shape}")

        # Plot Heatmap
        st.write("### Heatmap of Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(selected_df.corr(), annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Plot Multiple Scatterplots
        st.write("### Scatterplots for Numeric Features")
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=selected_features[i], y=selected_features[j], hue=target_col, data=selected_df,
                                palette='viridis', ax=ax)
                ax.set_title(f'Scatterplot of {selected_features[i]} vs {selected_features[j]}')
                st.pyplot(fig)





        # Plot Pairplot
        st.write("### Pairplot of Selected Features with Target Column")
        pairplot_fig = sns.pairplot(selected_df, hue=target_col, palette='husl')
        st.pyplot(pairplot_fig)

        X = selected_df.drop(columns=[target_col], axis=1)  # Correct way
        Y = selected_df[target_col]

        X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.2 , random_state=2)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define Models and Hyperparameters
        models = {
            "Logistic Regression": (LogisticRegression(), {"C": [0.1, 1, 10], "max_iter": [100, 200]}),
            "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}),
            "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
            "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
            "Naive Bayes": (GaussianNB(), {}),  # No hyperparameters to tune
            "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [None, 10, 20]})
        }

        # Store Results
        results = []

        for model_name, (model, params) in models.items():
            st.write(f"### Training {model_name}...")
            if params:
                grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='f1')
                grid_search.fit(X_train, Y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model.fit(X_train, Y_train)

            # Predictions
            Y_train_pred = best_model.predict(X_train)
            Y_test_pred = best_model.predict(X_test)

            # Metrics Calculation
            train_f1 = f1_score(Y_train, Y_train_pred, average='macro')
            train_recall = recall_score(Y_train, Y_train_pred, average='macro')
            test_f1 = f1_score(Y_test, Y_test_pred, average='macro')
            test_recall = recall_score(Y_test, Y_test_pred, average='macro')
            test_precision = precision_score(Y_test, Y_test_pred, average='macro')
            accuracy = accuracy_score(Y_test, Y_test_pred)

            results.append([model_name, train_f1, train_recall, test_f1, test_recall, test_precision, accuracy])

        # Create DataFrame for Results
        results_df = pd.DataFrame(results, columns=["Model_Name", "Train_f1", "Train_recall", "Test_f1", "Test_recall",
                                                    "Test_precision", "Accuracy_Score"])
        st.write("### Model Performance Table")
        st.write(results_df)

        # Find the Best Model
        best_model_name = results_df.loc[results_df['Accuracy_Score'].idxmax(), 'Model_Name']
        st.write(f"### Best Performing Model: {best_model_name}")

        # Plot Metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.set_index("Model_Name")[
            ["Train_f1", "Test_f1", "Test_recall", "Test_precision", "Accuracy_Score"]].plot(kind='bar', ax=ax)
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        best_model_row = results_df.loc[results_df['Accuracy_Score'].idxmax()]
        best_model_name = best_model_row['Model_Name']

        st.write(f"### 🏆 Best Performing Model: {best_model_name}")
        st.write(best_model_row)  # Display all performance metrics of the best model











import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from transformers import pipeline
import re
import nltk
from wordcloud import WordCloud

nltk.download('stopwords')
from nltk.corpus import stopwords

# Set up YouTube API
API_KEY = "AIzaSyBCtLCs14bng_B-TtlBOFoZNifmzeoysXg"  # Replace with your YouTube API Key
youtube = build("youtube", "v3", developerKey=API_KEY)

# Load Sentiment Model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# Function to extract video comments
def get_youtube_comments(video_url):
    video_id = re.search(r"v=([a-zA-Z0-9_-]+)", video_url).group(1)
    comments = []
    response = youtube.commentThreads().list(
        part="snippet", videoId=video_id, textFormat="plainText", maxResults=100
    ).execute()
    for item in response["items"]:
        comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    return comments


# Function to analyze sentiment
def analyze_sentiments(comments):
    results = sentiment_pipeline(comments)
    sentiments = [result['label'] for result in results]
    return sentiments


# Function to summarize key negative themes
def generate_summary(negative_comments):
    summarizer = pipeline("summarization")
    text = " ".join(negative_comments[:5])  # Use first 5 negative comments
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# Streamlit UI
st.title("YouTube Negative Comment Analyzer & Summarizer")
st.write("Paste a YouTube video URL to analyze negative comments and summarize key issues.")

video_url = st.text_input("Enter YouTube Video URL:")
if st.button("Analyze Comments"):
    with st.spinner("Fetching and analyzing comments..."):
        comments = get_youtube_comments(video_url)
        sentiments = analyze_sentiments(comments)

        df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})
        negative_comments = df[df["Sentiment"] == "NEGATIVE"]["Comment"].tolist()
        summary = generate_summary(negative_comments) if negative_comments else "No negative comments found."

        # Display results
        st.subheader("Sentiment Analysis")
        st.dataframe(df)

        # Pie Chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
        st.pyplot(fig)

        # Word Cloud for Negative Comments
        st.subheader("Negative Comment Word Cloud")
        stop_words = set(stopwords.words('english'))
        negative_text = " ".join(negative_comments)
        wordcloud = WordCloud(width=800, height=400, stopwords=stop_words, background_color="black").generate(
            negative_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Summary
        st.subheader("Summary of Negative Comments")
        st.write(summary)
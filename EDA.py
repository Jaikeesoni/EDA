import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def plot_histogram(df, column):
    fig = px.histogram(df, x=column, marginal="box", hover_data=df.columns)
    st.plotly_chart(fig)

def plot_boxplot(df, column):
    fig = px.box(df, y=column)
    st.plotly_chart(fig)

def plot_scatter(df, x_column, y_column):
    fig = px.scatter(df, x=x_column, y=y_column, trendline="ols")
    st.plotly_chart(fig)

def plot_correlation_heatmap(df, columns):
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig)

def plot_bar_chart(df, column):
    value_counts = df[column].value_counts()
    fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': column, 'y': 'Count'})
    st.plotly_chart(fig)

def plot_pie_chart(df, column):
    value_counts = df[column].value_counts()
    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Distribution of {column}')
    st.plotly_chart(fig)

def calculate_statistics(df, column):
    stats_df = df[column].describe()
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    stats_df['skewness'] = skewness
    stats_df['kurtosis'] = kurtosis
    return stats_df

def perform_hypothesis_test(df, column1, column2):
    _, p_value = stats.ttest_ind(df[column1], df[column2])
    return p_value

def main():
    st.title("Advanced Exploratory Data Analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader("Data Preview")
            st.write(df.head())
            
            st.subheader("Data Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            numeric_columns = get_numeric_columns(df)
            categorical_columns = get_categorical_columns(df)
            
            st.subheader("Univariate Analysis")
            selected_column = st.selectbox("Select a column for univariate analysis", df.columns)
            
            if selected_column in numeric_columns:
                st.write("Histogram and Box Plot")
                plot_histogram(df, selected_column)
                plot_boxplot(df, selected_column)
                
                st.write("Statistics")
                st.write(calculate_statistics(df, selected_column))
            else:
                st.write("Bar Chart")
                plot_bar_chart(df, selected_column)
                
                st.write("Pie Chart")
                plot_pie_chart(df, selected_column)
            
            st.subheader("Bivariate Analysis")
            x_column = st.selectbox("Select X-axis column", numeric_columns)
            y_column = st.selectbox("Select Y-axis column", numeric_columns)
            
            if x_column != y_column:
                st.write("Scatter Plot")
                plot_scatter(df, x_column, y_column)
                
                p_value = perform_hypothesis_test(df, x_column, y_column)
                st.write(f"P-value of t-test: {p_value:.4f}")
                if p_value < 0.05:
                    st.write("There is a statistically significant difference between the two variables.")
                else:
                    st.write("There is no statistically significant difference between the two variables.")
            
            st.subheader("Correlation Analysis")
            if len(numeric_columns) > 1:
                st.write("Correlation Heatmap")
                plot_correlation_heatmap(df, numeric_columns)
            else:
                st.write("Not enough numeric columns for correlation analysis.")
            
            st.subheader("Additional Insights")
            st.write(f"- The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.write(f"- There are {len(numeric_columns)} numeric columns and {len(categorical_columns)} categorical columns.")
            st.write(f"- Missing values: {df.isnull().sum().sum()}")
            st.write(f"- Duplicate rows: {df.duplicated().sum()}")
            
            st.subheader("Recommendations")
            st.write("- Handle missing values through imputation or removal.")
            st.write("- Remove duplicate rows if they are not intentional.")
            st.write("- For machine learning tasks, consider encoding categorical variables and scaling numeric variables.")
            st.write("- Investigate outliers in numeric columns and decide on appropriate treatment.")
            st.write("- For highly correlated features, consider feature selection or dimensionality reduction techniques.")
            st.write("- Analyze the distribution of target variables (if any) and consider transformations if needed.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
@st.cache_data
def load_data():
    # For this example, we'll create a sample dataset. In a real scenario, you'd load your actual data here.
    data = {
        'Make': ['Maruti', 'Hyundai', 'Tata', 'Mahindra', 'Honda'] * 20,
        'Model': ['Swift', 'i20', 'Nexon', 'XUV300', 'City'] * 20,
        'Year': np.random.randint(2015, 2024, 100),
        'Sales': np.random.randint(5000, 50000, 100),
        'Price': np.random.randint(500000, 2000000, 100),
        'Fuel_Type': np.random.choice(['Petrol', 'Diesel', 'Electric', 'CNG'], 100),
        'Body_Type': np.random.choice(['Hatchback', 'Sedan', 'SUV', 'MUV'], 100),
    }
    return pd.DataFrame(data)

# Function for data exploration
def explore_data(df):
    st.subheader("Data Overview")
    st.write(df.head())
    st.write(f"Shape of the dataset: {df.shape}")
    
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
    
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

# Function for visualizations
def create_visualizations(df):
    st.subheader("Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Sales'], kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Top 10 Models by Sales")
    top_models = df.groupby('Model')['Sales'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    top_models.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Sales by Fuel Type")
    fig, ax = plt.subplots()
    df.groupby('Fuel_Type')['Sales'].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Average Price by Body Type")
    fig, ax = plt.subplots()
    sns.barplot(x='Body_Type', y='Price', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Sales Trend Over Years")
    yearly_sales = df.groupby('Year')['Sales'].sum()
    fig, ax = plt.subplots()
    yearly_sales.plot(kind='line', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Price vs Sales Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Price', y='Sales', hue='Make', data=df, ax=ax)
    st.pyplot(fig)

# Function for insights and recommendations
def generate_insights_and_recommendations(df):
    st.subheader("Insights")
    insights = [
        f"The dataset contains sales data for {df['Year'].nunique()} years, from {df['Year'].min()} to {df['Year'].max()}.",
        f"The top-selling car model is {df.groupby('Model')['Sales'].sum().idxmax()} with {df.groupby('Model')['Sales'].sum().max()} units sold.",
        f"The most common fuel type is {df['Fuel_Type'].mode()[0]}.",
        f"The average car price in the dataset is ₹{df['Price'].mean():,.0f}.",
        f"The body type with the highest average price is {df.groupby('Body_Type')['Price'].mean().idxmax()}.",
        f"The year with the highest total sales was {df.groupby('Year')['Sales'].sum().idxmax()}."
    ]
    for insight in insights:
        st.write(f"- {insight}")
    
    st.subheader("Recommendations")
    recommendations = [
        "Analyze the factors contributing to the success of the top-selling models and apply these insights to other models.",
        "Consider expanding the range of electric and CNG vehicles if their sales show a positive trend.",
        "Investigate the correlation between price and sales to optimize pricing strategies.",
        "Focus marketing efforts on the body types that show the highest sales and profitability.",
        "Conduct a deeper analysis of yearly sales trends to identify any patterns or external factors affecting sales.",
        "Explore the possibility of introducing new models in the most popular segments."
    ]
    for recommendation in recommendations:
        st.write(f"- {recommendation}")

# Main function
def main():
    st.title("India Car Sales Exploratory Data Analysis")
    
    # Introduction
    st.header("Introduction")
    st.write("""
    Welcome to the India Car Sales Exploratory Data Analysis (EDA) App! This tool provides insights into car sales data in India, including:
    - Data exploration of sales, prices, and car characteristics
    - Visualizations of sales trends, popular models, and market segments
    - Insights and recommendations based on the analysis
    
    Let's dive into the data and discover interesting patterns in the Indian automotive market!
    """)
    
    # Load data
    df = load_data()
    
    # Data Exploration
    st.header("Data Exploration")
    explore_data(df)
    
    # Visualizations
    st.header("Visualizations")
    create_visualizations(df)
    
    # Insights and Recommendations
    st.header("Insights and Recommendations")
    generate_insights_and_recommendations(df)
    
    # Conclusion
    st.header("Conclusion")
    st.write("""
    This exploratory data analysis of India's car sales data has provided valuable insights into the automotive market trends. We've examined sales distributions, popular models, fuel type preferences, pricing strategies, and yearly trends.

    Key takeaways include:
    - Identification of top-selling models and manufacturers
    - Understanding of market preferences in terms of fuel types and body styles
    - Recognition of pricing trends across different segments
    - Insights into yearly sales patterns

    These findings can be used to inform business strategies, product development, and marketing campaigns in the Indian automotive industry. However, it's important to note that this analysis is based on a sample dataset and may not reflect the entire market accurately. For more comprehensive insights, consider incorporating additional data sources and conducting more detailed statistical analyses.

    Next steps could include:
    - Deeper analysis of regional sales patterns
    - Investigation of the impact of economic factors on car sales
    - Competitor analysis and market positioning studies
    - Consumer preference surveys to complement sales data

    Thank you for using the India Car Sales EDA App!
    """)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd

# App Title
st.title("Interactive Data Explorer")

# File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Load Dataset
    df = pd.read_csv(uploaded_file)

    # Dataset Overview
    st.header("Dataset Overview")
    st.write(f"Shape of the dataset: {df.shape}")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    # Show Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Select Columns for Exploration
    st.sidebar.header("Exploration Options")
    all_columns = df.columns.tolist()

    # Display Columns
    selected_columns = st.sidebar.multiselect("Select columns to display", all_columns, default=all_columns)
    if selected_columns:
        st.subheader("Selected Columns")
        st.dataframe(df[selected_columns])

    # Correlation Matrix
    if st.sidebar.checkbox("Show Correlation Matrix"):
        st.subheader("Correlation Matrix")
        st.write(df.corr())

    # Plot Options
    st.sidebar.header("Visualization Options")
    plot_type = st.sidebar.selectbox("Select plot type", ["None", "Histogram", "Bar Chart", "Line Chart", "Scatter Plot"])

    if plot_type != "None":
        st.subheader(f"{plot_type} of Selected Columns")

        if plot_type == "Histogram":
            column = st.sidebar.selectbox("Select column for Histogram", all_columns)
            bins = st.sidebar.slider("Number of bins", min_value=5, max_value=50, value=10)
            st.bar_chart(df[column].value_counts(bins=bins))

        elif plot_type in ["Bar Chart", "Line Chart"]:
            column = st.sidebar.selectbox(f"Select column for {plot_type}", all_columns)
            if plot_type == "Bar Chart":
                st.bar_chart(df[column])
            else:
                st.line_chart(df[column])

        elif plot_type == "Scatter Plot":
            x_col = st.sidebar.selectbox("X-axis", all_columns)
            y_col = st.sidebar.selectbox("Y-axis", all_columns)
            st.scatter_chart(df[[x_col, y_col]])
else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.caption("Built with ❤️ using Streamlit")

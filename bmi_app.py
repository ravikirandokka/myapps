import streamlit as st

# App Title
st.title("BMI Calculator")

# Input Fields
height = st.number_input("Enter your height (in cm):", min_value=0.0, format="%.2f")
weight = st.number_input("Enter your weight (in kg):", min_value=0.0, format="%.2f")

# Calculate BMI
if height > 0 and weight > 0:
    height_m = height / 100  # Convert height to meters
    bmi = weight / (height_m ** 2)
    st.subheader(f"Your BMI is: {bmi:.2f}")
    
    # Display BMI Category
    if bmi < 18.5:
        st.warning("You are underweight.")
    elif 18.5 <= bmi < 24.9:
        st.success("You have a normal weight.")
    elif 25 <= bmi < 29.9:
        st.warning("You are overweight.")
    else:
        st.error("You are obese.")
else:
    st.info("Please enter valid height and weight.")

# Footer
st.caption("Built with ❤️ using Streamlit")

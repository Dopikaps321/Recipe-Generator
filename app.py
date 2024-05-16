import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the saved MLP model
model_path = "C:/Users/Alaxo joy/OneDrive/Desktop/New folder/trained_mlp_model.pkl"
mlp_model = joblib.load(model_path)

# Load the dataset
recipes_df = pd.read_csv("Mini_Cleaned_Recipe_dataset.csv")

# Vectorize the ingredients
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(recipes_df['Ingredients'])

# Encode the recipe names
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(recipes_df['Recipe_Name'])

# Function to preprocess the data
def preprocess_data(data):
    return data.lower()  # Convert text to lowercase

# Function to generate recipe based on user-provided ingredients
def generate_recipe(ingredients, mlp_model, vectorizer, label_encoder):
    processed_ingredients = preprocess_data(", ".join(ingredients))
    user_input = vectorizer.transform([processed_ingredients])
    predicted_label = mlp_model.predict(user_input)
    predicted_recipe = label_encoder.inverse_transform(predicted_label)[0]
    instructions = recipes_df.loc[recipes_df['Recipe_Name'] == predicted_recipe, 'Instruction'].values[0]
    # Add a space between each character in words
    instruction_paragraph = ' '.join([' '.join(word) for sentence in instructions for word in sentence.split()])
    return predicted_recipe, instruction_paragraph

# Streamlit UI
st.title("Recipe Generator")

ingredients = st.text_input("Enter ingredients separated by comma (e.g., tomato, onion, garlic):")
if st.button("Generate Recipe"):
    if ingredients:
        predicted_recipe, instruction_paragraph = generate_recipe(ingredients.split(","), mlp_model, vectorizer, label_encoder)
        st.subheader("Recipe Name:")
        st.write(predicted_recipe)
        st.subheader("Instructions:")
        st.write(instruction_paragraph)
    else:
        st.warning("Please enter some ingredients.")


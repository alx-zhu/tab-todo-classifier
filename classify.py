import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_relevance_score(tab_name, task_name):
    prompt = f"""
    Score how relevant the browser tab titled "{tab_name}" is to the task "{task_name}".
    
    Respond with a JSON object containing:
    - score: (float between 0 and 1, where 1.0 means extremely relevant and 0.0 means completely irrelevant)
    
    Return only valid JSON.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that scores relevance between tabs and tasks. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content.strip())
        return result['score'], result.get('explanation')
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def main():
    st.title("Tab-Task Relevance Scorer")
    
    # Input fields
    tab_name = st.text_input("Enter the tab name:")
    task_name = st.text_input("Enter the task name:")
    
    if st.button("Get Relevance Score"):
        if tab_name and task_name:
            score, explanation = get_relevance_score(tab_name, task_name)
            if score is not None:
                st.success(f"Relevance Score: {score:.2f}")
                
                # Visual representation of the score
                st.progress(score)
                if explanation:
                    st.write(f"Explanation: {explanation}")
        else:
            st.warning("Please enter both tab name and task name.")

if __name__ == "__main__":
    main()

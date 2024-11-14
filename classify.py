import streamlit as st
import openai
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

"""
espn.com\n
drive.google.com\n
gmail.com\n
openai.blog\n
medium.com\n
"""

"""
apply for jobs\n
research ai\n
"""

def get_relevance_score(tab_name, task_name, prev_score=None, prev_explanation=None):
    feedback_injection = (f"""
    Previously another assistant gave a score of {prev_score} with the following explanation: {prev_explanation}.
    Use this as context for your evaluation, but adjust as needed based on your own judgment.
    If your score differs from the previous one, briefly justify your score and explain your reasoning.
    """) if prev_score is not None and prev_explanation else ""

    prompt = f"""
    Score how relevant the browser tab titled "{tab_name}" is to the task "{task_name}". 
    {feedback_injection}
    
    Respond with a JSON object containing:
    - score: (float between 0 and 1, where 1.0 means extremely relevant and 0.0 means completely irrelevant)
    - explanation: (brief explanation of the score)
    
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
    
def calculate_average_relevance_score(tab_name, task_name, n=4, threads=4):
    scores = []
    
    # Use ThreadPoolExecutor to make repeated parallel calls for the same pair
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(get_relevance_score, tab_name, task_name) for _ in range(n)]
        
        for future in as_completed(futures):
            score, _ = future.result()
            if score is not None:
                scores.append(score)

    # Check if scores were collected successfully
    if scores:
        # Calculate the average of the normalized scores as the final relevance score
        final_score = np.mean(scores)
        return final_score
    else:
        return None

def calculate_relevance_score_with_feedback(tab_name, task_name, repeat=3):
    prev_score = None
    prev_explanation = None
    
    for _ in range(repeat):
        score, explanation = get_relevance_score(tab_name, task_name, prev_score, prev_explanation)
        if score is not None:
            prev_score = score
            prev_explanation = explanation
        else:
            break
    
    return prev_score
    

def tab_relevance_page(score_fn):
    st.title("Tab-Task Relevance Scorer")

    # Input fields for lists of tabs and tasks
    tabs = st.text_area("Enter tab names (one per line):")
    tasks = st.text_area("Enter task names (one per line):")

    if st.button("Get Relevance Scores"):
        # Split the input into lists, trimming whitespace
        tab_list = [tab.strip() for tab in tabs.splitlines() if tab.strip()]
        task_list = [task.strip() for task in tasks.splitlines() if task.strip()]

        if tab_list and task_list:
            # Initialize a DataFrame to store scores, with tasks as rows and tabs as columns
            scores_df = pd.DataFrame(index=tab_list, columns=task_list)

            # Fill the DataFrame with relevance scores
            for task_name in task_list:
                for tab_name in tab_list:
                    score = score_fn(tab_name, task_name)
                    if score is not None:
                        scores_df.at[tab_name, task_name] = f"{score:.2f}"

            # Display the DataFrame as a grid
            st.write("### Tab-Task Relevance Scores")
            st.dataframe(scores_df)
        else:
            st.warning("Please enter at least one tab and one task.")

# determine where a tab history diverges
# normalizing scores
# incorporating other data like screenshots (richer context)
# determine if a context changed (within a history, or even within a task)

if __name__ == "__main__":
    # tab_relevance_page(calculate_relevance_score_with_feedback)
    tab_relevance_page(calculate_average_relevance_score)

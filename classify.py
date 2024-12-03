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

###################################
######## GENERIC FUNCTIONS ########
###################################

# Relevance function must return a tuple with the relevance score and explanation
def generic_average_relevance_score(relevance_fn, args: list, n=4, threads=4):
    scores = []
    
    # Use ThreadPoolExecutor to make repeated parallel calls for the same pair
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(relevance_fn, *args) for _ in range(n)]
        
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

# Relevance function must return a tuple with the relevance score and explanation. Must also accept prev_score and prev_explanation as arguments.
def generic_feedback_relevance_score(relevance_fn, args: list, repeat=3):
    prev_score = None
    prev_explanation = None
    
    for _ in range(repeat):
        score, explanation = relevance_fn(*args, prev_score=prev_score, prev_explanation=prev_explanation)
        if score is not None:
            prev_score = score
            prev_explanation = explanation
        else:
            break
    
    return prev_score

#####################################
######## GET_SCORE FUNCTIONS ########
#####################################

def get_score_tab_to_task(tab_name, task_name, prev_score=None, prev_explanation=None):
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
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tab_to_task_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "description": "(float between 0 and 1, where 1.0 means extremely relevant and 0.0 means completely irrelevant)",
                                "type": "number",
                            },
                            "explanation": {
                                "description": "(brief explanation of the score)",
                                "type": "string",
                            },
                            "additionalProperties": False
                        }
                    }
                }
            }
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content.strip())
        print(result)
        return result['score'], result.get('explanation')
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def get_score_tab_to_history(tab_name, history, prev_score=None, prev_explanation=None):
    feedback_injection = (f"""
    Previously another assistant gave a score of {prev_score} with the following explanation: {prev_explanation}.
    Use this as context for your evaluation, but adjust as needed based on your own judgment.
    If your score differs from the previous one, briefly justify your score and explain your reasoning.
    """) if prev_score is not None and prev_explanation else ""

    prompt = f"""
    The user has been working on a task and has visited the following tabs in order: {history}.
    Score how likely the user is still working on the previous task by determining how likely 
    the browser tab titled "{tab_name}" has diverged from to the provided tab history {history}.
    {feedback_injection}

    Respond with a JSON object containing:
    - score: float between 0 and 1, where 0.0 means the tab has completely diverged from the tab history (or that it is highly likely that a new task has started), 
      and 1.0 means the tab is extremely relevant to the tab history (or that it is highly likely that the user is still working on the same task).
    - explanation: brief explanation of the score
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a tab classification expert that scores how relevant a tab is to a previous tab history.\
                 You are tasked with helping score how likely it is that a tab belongs to a provided history. Your highest priority goal is to \
                 help a user understand if the old task has continued (higher score) or if a new task has begun (lower score). Always respond with valid JSON."},
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

def average_tab_to_task_relevance(tab_name, task_name):
    return generic_average_relevance_score(get_score_tab_to_task, [tab_name, task_name])

def feedback_tab_to_task_relevance(tab_name, task_name):
    return generic_feedback_relevance_score(get_score_tab_to_task, [tab_name, task_name])

def average_tab_to_history_relevance(tab_name, history):
    return generic_average_relevance_score(get_score_tab_to_history, [tab_name, history])

def feedback_tab_to_history_relevance(tab_name, history):
    return generic_feedback_relevance_score(get_score_tab_to_history, [tab_name, history])

def tab_relevance_page(tab_to_task_fn, tab_to_history_fn):
    st.title("Tab-Task Relevance Scorer")

    st.subheader("Tab to Task Relevance")

    # Input fields for lists of tabs and tasks
    tabs = st.text_area("Enter tab names (one per line):", value="espn.com\ndrive.google.com\ngmail.com\nopenai.blog\nmedium.com\n")
    tasks = st.text_area("Enter task names (one per line):", value="apply for jobs\nresearch ai\n")

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
                    score = tab_to_task_fn(tab_name, task_name)
                    if score is not None:
                        scores_df.at[tab_name, task_name] = f"{score:.2f}"

            # Display the DataFrame as a grid
            st.write("### Tab-Task Relevance Scores")
            st.dataframe(scores_df)
        else:
            st.warning("Please enter at least one tab and one task.")
        
    st.subheader("Tab to History Relevance")
    history = st.text_area("Enter tab history (one per line, top is most recent):", value="OpenAI: GPT Models Overview\nResearch Paper: Transformers Explained\nMIT Technology Review: Latest AI Trends\nGoogle Drive\nGoogle Scholar: AI Ethics and Bias Papers\n")
    tab = st.text_input("Enter tab name:", "Twitter: AI Conference 2024 Updates")

    if st.button("Get Relevance Score"):
        # Split the input into a list, trimming whitespace
        history_list = [tab.strip() for tab in history.splitlines() if tab.strip()]

        if history_list and tab:
            # Get the relevance score
            score = tab_to_history_fn(tab, history_list)
            if score is not None:
                st.write(f"Tab-History Relevance Score: {score:.2f}")
                st.progress(score)
        else:
            st.warning("Please enter a tab and a tab history.")

if __name__ == "__main__":
    tab_relevance_page(average_tab_to_task_relevance, average_tab_to_history_relevance)

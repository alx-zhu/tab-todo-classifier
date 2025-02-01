import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from scoring.tab_scraper import fetch_tab_content

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

##################################
## TAB CLASSIFICATION FUNCTIONS ##
##################################

# Tournament based task to tab pairing
def pick_from_two_tasks(task1, task2, tab):
    prompt = f"""
        You are an expert task classifier tasked with picking which task is most relevant to a given tab.
        You are given this tab to classify tasks for: {tab}. Given the following two tasks, '1: {task1}' and '2: {task2},    
        select the task that is most relevant to the tab. 
        
        Respond with a JSON object containing:
        - id: the id of the selected task (1 or 2)
        - explanation: (brief explanation of why you selected the task)
    
        Return only valid JSON.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert task classifier tasked with picking which task is most relevant to a given tab. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "best_task_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "description": "the id of the selected task (1 or 2)",
                                "type": "number",
                            },
                            "explanation": {
                                "description": "(brief explanation of why you selected the task)",
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
        id, explanation = result['id'], result['explanation']
        if id < 1 or id > 2:
            raise ValueError("The id must be 1 or 2")
        
        return id, explanation
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None
    

def pick_from_many_tasks(tasks: list, tab: str) -> Tuple[int, str | None]:
    n = len(tasks)
    if n < 2:
        return tasks[0], "Only one task available"
    
    task_map = {i: task for i, task in enumerate(tasks)}
    prompt = f"""
        You are an expert task classifier tasked with picking which task is most relevant to a given tab.
        You are given this tab to classify tasks for: {tab}. Given the following map of id to task: {task_map},    
        select the task that is most relevant to the tab. 
        
        Respond with a JSON object containing:
        - id: the id of the selected task (number from 0 to {n-1})
        - explanation: (explanation of why you selected the task for the specific tab. Make sure the explanation is specific to the tab and the chosen task.)
    
        Return only valid JSON.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert task classifier tasked with picking which task is most relevant to a given tab. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "best_task_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "description": f"the id of the selected task (number from 0 to {n-1})",
                                "type": "number",
                            },
                            "explanation": {
                                "description": "(explanation of why you selected the task for the specific tab. Make sure the explanation is specific to the tab and the chosen task.)",
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
        id, explanation = result['id'], result['explanation']
        if id < 0 or id >= n:
            raise ValueError(f"The id must be from 0 to {n-1} (size of the task list)")
        
        return task_map[id], explanation
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def pick_from_many_tasks_with_content(tasks: list, tab_url: str, tab_content: str) -> Tuple[int, str | None]:
    n = len(tasks)
    if n < 2:
        return tasks[0], "Only one task available"
    
    task_map = {i: task for i, task in enumerate(tasks)}
    prompt = f"""
        You are an expert task classifier tasked with picking which task is most relevant to a given tab.
        You are given this tab to classify tasks for: {tab_url}. 
        The tab content is as follows: {tab_content}
        
        Given the following map of id to task: {task_map}, select the task that is most relevant to the tab, using the contents of the tab to guide your decision. 
        
        Respond with a JSON object containing:
        - id: the id of the selected task (number from 0 to {n-1})
        - explanation: (explanation of why you selected the task for the specific tab. Make sure the explanation is specific to the tab and the chosen task.)
    
        Return only valid JSON.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert task classifier tasked with picking which task is most relevant to a given tab. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "best_task_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "description": f"the id of the selected task (number from 0 to {n-1})",
                                "type": "number",
                            },
                            "explanation": {
                                "description": "(explanation of why you selected the task for the specific tab. Make sure the explanation is specific to the tab and the chosen task.)",
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
        id, explanation = result['id'], result['explanation']
        if id < 0 or id >= n:
            raise ValueError(f"The id must be from 0 to {n-1} (size of the task list)")
        
        return task_map[id], explanation
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

##################################
###### TOURNAMENT FUNCTIONS ######
##################################
def pick_from_many_tasks_tournament(tasks: list, tab: str, tasks_per_round=None, threads=4):
    """
    Tournament based tab classification, where tasks compete in rounds to be selected. The tournament continues until only one task remains.
    `tasks_per_round` determines how many tasks compete in each round. If None, all tasks compete in the first round.

    Args:
        tasks (list): List of tasks to classify.
        tab (str): The tab input.
        tasks_per_round (int, optional): Number of tasks to compete in each round. Defaults to None.
        threads (int, optional): Number of threads to use for parallel processing. Defaults to 4.

    Returns:
        tuple: The selected task and its explanation.
    """
    n = len(tasks)
    if n < 2:
        return None, None
    
    if not tasks_per_round or tasks_per_round >= n:
        return pick_from_many_tasks(tasks, tab)
    if tasks_per_round < 2:
        raise ValueError("tasks_per_round must be at least 2")
    
    curr_tasks = tasks
    while len(curr_tasks) > 1:
        new_tasks = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(0, len(curr_tasks), tasks_per_round):
                futures.append(executor.submit(pick_from_many_tasks, curr_tasks[i:i+tasks_per_round], tab))
            
            for future in as_completed(futures):
                task, _ = future.result()
                if task is not None:
                    new_tasks.append(task)
        curr_tasks = new_tasks

    if not curr_tasks:
        raise ValueError("No task selected")

    return curr_tasks[0], None

def pick_from_many_tasks_tournament_verbose(tasks, tab, tasks_per_round=None, threads=4):
    n = len(tasks)
    if n < 2:
        return None
    
    curr_tasks = [(task, "Initial Round") for task in tasks]
    if not tasks_per_round or tasks_per_round >= n or tasks_per_round < 2:
        tab, exp = pick_from_many_tasks(tasks, tab)
        return tab, [curr_tasks, [(tab, exp)]]
    if tasks_per_round < 2:
        raise ValueError("tasks_per_round must be at least 2")
    
    tournament = []
    while len(curr_tasks) > 1:
        new_tasks = []
        tournament.append(curr_tasks)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(0, len(curr_tasks), tasks_per_round):
                futures.append(executor.submit(pick_from_many_tasks, [task for (task, _) in curr_tasks[i:i+tasks_per_round]], tab))
            
            for future in as_completed(futures):
                task, explanation = future.result()
                if task is not None:
                    new_tasks.append((task, explanation))
        curr_tasks = new_tasks
    tournament.append(curr_tasks)

    if not curr_tasks:
        raise ValueError("No task selected")

    return curr_tasks[0], tournament

#################################
#### PARALLEL CLASSIFICATION ####
#################################

def generic_most_common_answer(classify_fn, args:list, n=4, threads=4):
    answers = []
    
    # Use ThreadPoolExecutor to make repeated parallel calls for the same pair
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(classify_fn, *args) for _ in range(n)]
        
        for future in as_completed(futures):
            answer, _ = future.result()
            if answer is not None:
                answers.append(answer)

    # Check if answers were collected successfully
    if answers:
        # Calculate the most common answer as the final classification
        final_answer = max(set(answers), key=answers.count)
        return final_answer
    else:
        return None
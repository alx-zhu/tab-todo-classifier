import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Tournament based task to tab pairing
def pick_from_two_tasks(task1, task2, tab):
    prompt = f"""
        You are an expert task classifier tabed with picking which task is most relevant to a given tab.
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
                {"role": "system", "content": "You are an expert task classifier tabed with picking which task is most relevant to a given tab. Always respond with valid JSON."},
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
    

def pick_from_many_tasks(tasks, tab) -> Tuple[int, str | None]:
    n = len(tasks)
    if n < 2:
        return tasks[0], "Only one task available"
    
    task_map = {i: task for i, task in enumerate(tasks)}
    prompt = f"""
        You are an expert task classifier tabed with picking which task is most relevant to a given tab.
        You are given this tab to classify tasks for: {tab}. Given the following map of id to task: {task_map},    
        select the task that is most relevant to the tab. 
        
        Respond with a JSON object containing:
        - id: the id of the selected task (number from 0 to {n-1})
        - explanation: (brief explanation of why you selected the task)
    
        Return only valid JSON.
    """

    print(tasks)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert task classifier tabed with picking which task is most relevant to a given tab. Always respond with valid JSON."},
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
        if id < 0 or id >= n:
            raise ValueError(f"The id must be from 0 to {n-1} (size of the task list)")
        
        return task_map[id], explanation
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def pick_from_many_tasks_tournament(tasks, tab, tasks_per_round=None, threads=4):
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
    
    if not tasks_per_round or tasks_per_round >= n:
        return pick_from_many_tasks(tasks, tab)
    if tasks_per_round < 2:
        raise ValueError("tasks_per_round must be at least 2")
    
    tournament = []
    curr_tasks = [(task, "Initial Round") for task in tasks]
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
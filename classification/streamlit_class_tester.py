import streamlit as st
from classification.classification import pick_from_many_tasks_tournament_verbose
import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Select a task
def select_task(i):
    st.session_state.selected_task = i
    st.session_state.task_fields = [[field, value] for (field, value) in st.session_state.tasks[i].items()]

# Function to add a new task
def add_task(task_name):
    if task_name and task_name not in st.session_state.tasks:
        st.session_state.tasks.append({"name": task_name})

# Function to delete a task
def delete_task(i):
    if i < len(st.session_state.tasks):
        st.session_state.tasks.pop(i)
    if st.session_state.selected_task == i:
        st.session_state.selected_task = None
    st.rerun()

# Function to add a field to a task
def add_field():
    st.session_state.task_fields.append(["", ""])

# Save the selected task
def save_selected_task():
    if st.session_state.selected_task:
        task_name = st.session_state.selected_task
        task_fields = {}
        for [field_name, field_value] in st.session_state.task_fields:
            if field_name and field_value:
                task_fields[field_name] = field_value
        st.session_state.tasks[task_name] = task_fields

# Query ChatGPT API to generate similar tasks to the set of current tasks for more data.
def generate_similar_tasks(n=10):
    tasks = json.dumps(st.session_state.tasks)
    prompt = f"""Given the following tasks and their JSON fields, generate {n} similar tasks with the same JSON fields: {tasks}
        Be creative and generate a diverse set of tasks that a typical computer user would perform in a browser. You can generate some similar tasks
        but try to create new tasks as well. Tasks must have a name field.
        
        Return a JSON object containing:
        - tasks: a list of {n} generated tasks in JSON format. Each task must have a name field.

        Return only valid JSON.
        """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a dataset generator tasked with creating diverse task JSON objects to test a tab and task classifier. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generated_tasks_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "description": "a list of generated tasks in JSON format",
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name"],
                                    "properties": {
                                        "name": {
                                            "type": "string"
                                        }
                                    },
                                    "additionalProperties": True
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                }
            }
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content.strip())
        return result['tasks']
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

def sidebar():
    st.subheader("Manage Tasks")
    new_task_name = st.text_input("New Task Name")
    if st.button("Add Task"):
        add_task(new_task_name)

    if st.session_state.tasks:
        if st.button("Generate Tasks with AI"):
            similar_tasks = generate_similar_tasks()
            print(similar_tasks)
            st.session_state.tasks.extend(similar_tasks)

    st.divider()
    st.write("### Current Tasks")
    if not st.session_state.tasks:
        st.write("No tasks available. Add a new task to get started.")

    for i, task in enumerate(st.session_state.tasks):
        l, r = st.columns([3, 1])
        task_name = task["name"]
        with l:
            if st.button(task_name, key=f"select_{task_name}", use_container_width=True):
                select_task(i)
        with r:
            if st.button(f":wastebasket:", key=f"delete_{task_name}"):
                delete_task(i)
    
    if st.button("Get JSON"):
        st.json(st.session_state.tasks)
    
def edit_tasks_tab():
    if st.session_state.selected_task is not None:
        selected_task = st.session_state.selected_task

        for i, [field_name, field_value] in enumerate(st.session_state.task_fields):
            l, r, d = st.columns([5, 5, 1])
            with l:
                field_id = f"edit_field_name_{selected_task}_{field_name}_{i}"
                st.text_input(f"Field", value=field_name, key=field_id)
                st.session_state.task_fields[i][0] = st.session_state[field_id]
            with r:
                value_id = f"edit_field_value_{selected_task}_{field_name}_{i}"
                st.text_input(f"Value", value=field_value, key=value_id)
                st.session_state.task_fields[i][1] = st.session_state[value_id]
            with d:
                if st.button(f":wastebasket:", key=f"delete_field_{selected_task}_{field_name}_{i}"):
                    st.session_state.task_fields.pop(i)
                    st.rerun()
            
        # Add a new field
        if st.button("Add Field", key=f"add_field_{selected_task}"):
            add_field()
            st.rerun()

        if st.button("Save Task", key=f"save_task_{selected_task}"):
            save_selected_task()
            st.rerun()
            st.success("Task saved successfully")
        
        # Submit and display JSON
        st.write("### JSON Output")
        st.json(dict(st.session_state.task_fields))

    else:
        st.write("No task selected. Click on a task to edit its fields.")

def test_classifier_tab():
    tab_info = st.text_area("Tab name or JSON object")
    n = st.text_input("Number of tabs to compare", value=2)
    if st.button("Classify Tab"):
        if not n.isnumeric():
            st.error("Please enter a valid number of tabs to compare.")
            return
        else:
            tab, tournament = pick_from_many_tasks_tournament_verbose(st.session_state.tasks, tab_info, int(n))
            st.write(f"Selected tab: {tab}")
            st.write("### Tournament Rounds")
            for i, r in enumerate(tournament):
                st.write(f"Round: {i}")
                st.write([{"task": task["name"], "explanation": exp} for (task, exp) in r])

def test_one_to_many_main_page():
    # Main UI
    st.title("CRUD Application for Tasks with JSON Fields")

    # Add a new task
    with st.sidebar:
        sidebar()

    [edit_tasks, test_classifier] = st.tabs(["Edit Tasks", "Test Classifier"])

    with edit_tasks:
        edit_tasks_tab()
    
    with test_classifier:
        test_classifier_tab()

def initialize():
    # read example tasks from example.json
    if "tasks" not in st.session_state or 'selected_task' not in st.session_state:
        st.session_state.selected_task = None
        st.session_state.task_fields = []
        with open('example.json') as f:
            st.session_state.tasks = json.load(f)

if __name__ == '__main__':
    initialize()
    test_one_to_many_main_page()
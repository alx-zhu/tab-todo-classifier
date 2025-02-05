import openai
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
from collections import defaultdict
from scoring.tab_scraper import fetch_tab_content

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

def ranked_choice_voting(ballots, candidates):
    """
    Perform ranked-choice voting.
    :param ballots: List of lists, where each sublist is a ranked list of candidates.
    :param candidates: List of all possible candidates.
    :return: The winning candidate.
    """
    candidate_pool = set(candidates)
    
    while True:
        vote_counts = defaultdict(int)
        
        # Count first-choice votes
        for ballot in ballots:
            for candidate in ballot:
                if candidate in candidate_pool:
                    vote_counts[candidate] += 1
                    break  # Count only the highest-ranked valid candidate
        
        # Check if a candidate has more than 50% of the votes
        total_votes = sum(vote_counts.values())
        for candidate, count in vote_counts.items():
            if count > total_votes / 2:
                return candidate
        
        # Find the candidate(s) with the fewest votes
        min_votes = min(vote_counts.values())
        eliminated_candidates = {c for c, v in vote_counts.items() if v == min_votes}
        
        # Remove eliminated candidates
        candidate_pool -= eliminated_candidates
        
        # If only one candidate remains, they are the winner
        if len(candidate_pool) == 1:
            return candidate_pool.pop()
        
        # If there's a tie among all remaining candidates, return one arbitrarily
        if len(candidate_pool) == 0:
            return list(vote_counts.keys())[0]  # Return any candidate as a tie-breaker

def get_task_ranking(tab_url, task_names):
    """
    Use OpenAI API to rank tasks based on tab_url relevance.
    """

    tab_content = fetch_tab_content(tab_url)
    task_map = {i: task for i, task in enumerate(task_names)}
    
    prompt = f"""
    You are an expert task classifier tasked with ranking the given tasks based on how likely the tab is to belong to that task.
    You are given this tab to classify tasks for: {tab_url}. 
    The tab content is as follows: {tab_content}
    
    Given the following map of id to task: {task_map}, Rank the tasks in order of the tasks that the tab is most likely to belong to, using the contents of the tab to guide your decision.
    Provide a ranked list with the most relevant task first. 

    Make sure to use the context of the tab URL itself in your decisions, in combination with the tab content.
    
    Respond with a JSON object containing:
    - ranking: a ranked list of the task id's of the tasks in order of relevance to the tab. Must have length {len(task_names)}.
    - explanation: (explanation of your ranking. Make sure the explanation is specific to the tab and the ranked tasks.)

    Return only valid JSON.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ranking_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranking": {
                            "description": "a ranked list of the task id's of the tasks in order of relevance to the tab.",
                            "type": "array",
                            "items": {
                                "type": "integer"
                            },
                            "minItems": len(task_names),
                            "maxItems": len(task_names)
                        },
                        "explanation": {
                            "description": "(explanation of your ranking. Make sure the explanation is specific to the tab and the ranked tasks.)",
                            "type": "string"
                        },
                        "additionalProperties": False
                    },
                    "required": ["ranking", "explanation"]
                }
            }
        },
        temperature=0.8,
        messages=[{"role": "system", "content": "You are an expert task classifier tasked with ranking tasks based on relevance to a given tab. Always respond with valid JSON."},
                  {"role": "user", "content": prompt}],
    )
    
    ranked_tasks_str = response.choices[0].message.content.strip()
    ranked_tasks = json.loads(ranked_tasks_str)["ranking"]
    return ranked_tasks

def run_rcv(tab_url: str, task_names: list, num_voters: int = 10) -> Tuple[list, dict]:
    """
    Rank tasks based on their relevance to a given tab URL.
    :param tab_url: The URL of the tab to analyze.
    :param task_names: List of task names to rank.
    :return: Tuple containing the ranked task IDs and a dictionary with explanations.
    """

    with ThreadPoolExecutor() as executor:
        ballots = []
        for future in as_completed([executor.submit(get_task_ranking, tab_url, task_names) for _ in range(num_voters)]):
            try:
                ranking = future.result()
                ballots.append(ranking)
            except Exception as e:
                print(f"Error ranking: {e}")
    
    # Perform ranked-choice voting
    candidates = list(range(len(task_names)))
    winning_candidate = ranked_choice_voting(ballots, candidates)
    
    return task_names[winning_candidate], ballots

if __name__ == "__main__":
    tasks = [
        "Watch AI Course", # Comment out
        "Watch ML Course", # RCV most frequently selects Watch Online Course, if this is not added. Machine Learning is not recognized as the same as AI.
        "Research AI",
        "Write History Paper",
        "Apply to Jobs",
        "Watch Online Course",
        "Read News Articles",
        "Travel Abroad",
        "Shop for Laptop",
        "Review Work Documents",
        "Manage Personal Finances",
        "Edit Photos"
    ]

    winner, ballots = run_rcv("https://www.coursera.org/learn/machine-learning", tasks)
    print("Winner:", winner)
    print("Ballots:", ballots)

# Before adding Watch AI Course and Watch ML Course, the above link would turn out to be "Watch Online Course" every time.
# Seems like a machine-learning course was not recognized as "Research AI". It was the second vote almost every time.
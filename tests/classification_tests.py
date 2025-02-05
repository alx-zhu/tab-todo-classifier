from classification.classification import pick_from_many_tasks_tournament_with_content

from collections import Counter
import random

def test_task_picker_consistency(pick_function, tasks, tab_url, trials=10, tasks_per_round=None):
    """
    Runs multiple trials of the pick function and analyzes consistency.
    
    Args:
        pick_function (callable): The function to test.
        tasks (list): List of tasks to classify.
        tab_url (str): The tab URL input.
        trials (int): Number of times to run the test.
        tasks_per_round (int, optional): Number of tasks per round.
        threads (int, optional): Number of threads to use.
    
    Returns:
        dict: A dictionary containing frequency of each selected task and consistency metrics.
    """
    selection_counts = Counter()
    
    for _ in range(trials):
        selected_task, _ = pick_function(tasks, tab_url, tasks_per_round)
        if selected_task is not None:
            selection_counts[selected_task] += 1
    
    total_selections = sum(selection_counts.values())
    
    if total_selections == 0:
        return {"error": "No task was selected in any trial."}
    
    most_frequent_task = max(selection_counts, key=selection_counts.get)
    consistency_score = selection_counts[most_frequent_task] / total_selections
    
    return {
        "total_trials": trials,
        "total_selections": total_selections,
        "selection_counts": dict(selection_counts),
        "most_frequent_task": most_frequent_task,
        "consistency_score": consistency_score  # Percentage of times the most frequent task was picked
    }

tasks = [
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

# Run an individual unit test
def unit_test_classifier(i, tab_url, tasks, tasks_per_round=None):
    print(f"---------- Unit Test {i} ----------\n")
    print("Tab URL:", tab_url)
    print("Tasks:", tasks)
    results = test_task_picker_consistency(pick_from_many_tasks_tournament_with_content, tasks, tab_url, tasks_per_round=tasks_per_round)
    print("Results:", results)
    print("\n")

def run_unit_test_suite(suite_name, tab_urls, tasks, tasks_per_round=None):
    print(f"=====================================")
    print(f"{suite_name}: { "All" if not tasks_per_round else tasks_per_round} Tasks Per Round")
    print(f"=====================================")
    for i, tab_url in enumerate(tab_urls):
        unit_test_classifier(i, tab_url, tasks, tasks_per_round)
    print(f"{suite_name} completed.\n")

def research_ai_tests(tasks_per_round=None):
    research_ai_links = [
        "https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
        "https://www.coursera.org/learn/machine-learning",
        "https://www.coursera.org/learn/deep-learning-ai",
    ]
    run_unit_test_suite("Research AI", research_ai_links, tasks, tasks_per_round=tasks_per_round)

def write_history_paper_tests(tasks_per_round=None):
    write_history_paper_links = [
        "https://www.youtube.com/watch?v=_uk_6vfqwTA",
        "https://www.history.com/topics/world-war-ii",
    ]
    run_unit_test_suite("Write History Paper", write_history_paper_links, tasks, tasks_per_round=tasks_per_round)

def apply_to_jobs_tests(tasks_per_round=None):
    apply_to_jobs_links = [
        "https://jobs.ashbyhq.com/kleinerperkinsfellows/97a02c44-099d-490d-9353-136e7a25fb4c",
        "https://www.linkedin.com/jobs/",
        "https://www.indeed.com/",
        "https://job-boards.greenhouse.io/perplexityai/jobs/4537480007",
    ]
    run_unit_test_suite("Apply to Jobs", apply_to_jobs_links, tasks, tasks_per_round=tasks_per_round)


def watch_online_course_tests(tasks_per_round=None):
    watch_online_course_links = [
        "https://www.coursera.org/",
        "https://www.udacity.com/",
        "https://www.udemy.com/",
    ]
    run_unit_test_suite("Watch Online Course", watch_online_course_links, tasks, tasks_per_round=tasks_per_round)

def read_news_articles_tests(tasks_per_round=None):
    read_news_articles_links = [
        "https://www.nytimes.com/",
        "https://www.theguardian.com/us",
        "https://www.bbc.com/news",
    ]
    run_unit_test_suite("Read News Articles", read_news_articles_links, tasks, tasks_per_round=tasks_per_round)

def travel_abroad_tests(tasks_per_round=None):
    travel_abroad_links = [
        "https://wwoof.es/en/hosts?map.lat=40.9415270946638&map.lon=-3.71612548828125&map.zoom=7",
        "https://www.hostelworld.com/",
        "https://www.airbnb.com/",
    ]
    run_unit_test_suite("Travel Abroad", travel_abroad_links, tasks, tasks_per_round=tasks_per_round)

def shop_for_laptop_tests(tasks_per_round=None):
    shop_for_laptop_links = [
        "https://www.apple.com/shop/buy-mac/macbook-pro/13",
        "https://www.dell.com/en-us/shop/dell-laptops/sc/laptops",
    ]
    run_unit_test_suite("Shop for Laptop", shop_for_laptop_links, tasks, tasks_per_round=tasks_per_round)

def review_work_documents_tests(tasks_per_round=None):
    review_work_documents_links = [
        "https://drive.google.com",
        "https://docs.google.com",
    ]
    run_unit_test_suite("Review Work Documents", review_work_documents_links, tasks, tasks_per_round=tasks_per_round)

def manage_personal_finances_tests(tasks_per_round=None):
    manage_personal_finances_links = [
        "https://www.bankofamerica.com/",
        "https://www.chase.com/",
    ]
    run_unit_test_suite("Manage Personal Finances", manage_personal_finances_links, tasks, tasks_per_round=tasks_per_round)

def edit_photos_tests(tasks_per_round=None):
    edit_photos_links = [
        "https://www.canva.com/photo-editor/",
        "https://www.adobe.com/express/",
    ]
    run_unit_test_suite("Edit Photos", edit_photos_links, tasks, tasks_per_round=tasks_per_round)

def ambiguous_tests(tasks_per_round=None):
    ambiguous_links = [
        "https://mail.google.com",
        "https://calendar.google.com",
        "https://drive.google.com",
    ]
    run_unit_test_suite("Ambiguous", ambiguous_links, tasks, tasks_per_round=tasks_per_round)

def run_classification_unit_tests(tasks_per_round=None):
    print("Testing task picker consistency...")
    research_ai_tests(tasks_per_round=tasks_per_round)
    write_history_paper_tests(tasks_per_round=tasks_per_round)
    apply_to_jobs_tests(tasks_per_round=tasks_per_round)
    watch_online_course_tests(tasks_per_round=tasks_per_round)
    read_news_articles_tests(tasks_per_round=tasks_per_round)
    travel_abroad_tests(tasks_per_round=tasks_per_round)
    shop_for_laptop_tests(tasks_per_round=tasks_per_round)
    review_work_documents_tests(tasks_per_round=tasks_per_round)
    manage_personal_finances_tests(tasks_per_round=tasks_per_round)
    edit_photos_tests(tasks_per_round=tasks_per_round)
    ambiguous_tests(tasks_per_round=tasks_per_round)
    print("All tests completed.")

if __name__ == "__main__":
    # run_classification_unit_tests()
    # python -m tests.classification_tests > tests/classification_test_results.txt
    run_classification_unit_tests(tasks_per_round=2)
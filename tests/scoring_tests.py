import numpy as np
from scoring.scoring import average_tab_to_task_relevance_with_content
from scoring.tab_scraper import fetch_tab_content

def test_relevance_scoring(scoring_function, tab, task, num_trials=10):
    """
    Tests the consistency of a relevance scoring function by running it multiple times 
    and analyzing the variation in scores.
    
    Args:
        scoring_function (callable): The function that takes `tab` and `task` as input and returns a relevance score.
        tab (str): The tab input.
        task (str): The task input.
        num_trials (int): Number of times to run the scoring function.

    Returns:
        dict: Statistics on score consistency (mean, std deviation, min, max).
    """
    scores = [scoring_function(tab, task) for _ in range(num_trials)]
    
    return {
        "mean_score": np.mean(scores),
        "std_dev": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "score_range": np.max(scores) - np.min(scores),
    }


def test_relevance_scoring_with_content(scoring_function, tab_url, task, num_trials=10):
    """
    Tests the consistency of a relevance scoring function by running it multiple times 
    and analyzing the variation in scores. This function also includes the tab content 
    in the scoring function.

    Args:
        scoring_function (callable): The function that takes `tab` and `task` as input and returns a relevance score.
        tab (str): The tab input.
        task (str): The task input.
        num_trials (int): Number of times to run the scoring function.

    Returns:
        dict: Statistics on score consistency (mean, std deviation, min, max).
    """
    tab_content = fetch_tab_content(tab_url)
    scores = [scoring_function(tab_url, tab_content, task, temperature=0.8) for _ in range(num_trials)]
    
    return {
        "mean_score": np.mean(scores),
        "std_dev": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "score_range": np.max(scores) - np.min(scores),
    }


# Run an individual unit test
def unit_test_with_content(test_name, tab_url, task, index):
    print(f"---------- Unit Test {index} ----------\n")
    print("Test Name:", test_name)
    print("Tab URL:", tab_url)
    print("Task:", task)
    results = test_relevance_scoring_with_content(average_tab_to_task_relevance_with_content, tab_url, task)
    print("Results:", results)
    print("\n")

# Run a suite of unit tests, providing a name and a list of test pairs (test_name, tab_url, task)
def run_unit_test_suite(suite_name, test_pairs):
    print(f"=====================================")
    print(f"{suite_name}")
    print(f"=====================================")
    for i, (test_name, tab, task) in enumerate(test_pairs):
        unit_test_with_content(test_name, tab, task, i+1)
    print(f"{suite_name} completed.\n")

def high_relevance_tests():
    testing_pairs = [
                      ("High Relevance: AI Youtube x Research AI", "https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", "Research AI"),
                      ("High Relevance: Job Posting x Apply to Jobs", "https://jobs.ashbyhq.com/kleinerperkinsfellows/97a02c44-099d-490d-9353-136e7a25fb4c", "Apply to Jobs")
                    ]
    run_unit_test_suite("High Relevance Tests", testing_pairs)

def low_relevance_tests():
    testing_pairs = [
                     ("Low Relevance: AI Youtube x Apply to Jobs", "https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", "Apply to Jobs"),
                     ("Low Relevance: Job Posting x Research AI", "https://jobs.ashbyhq.com/kleinerperkinsfellows/97a02c44-099d-490d-9353-136e7a25fb4c", "Research AI"),
                     ("Low Relevance: Travel Abroad x Research AI", "https://wwoof.es/en/hosts?map.lat=40.9415270946638&map.lon=-3.71612548828125&map.zoom=7", "Research AI"),
                     ("Low Relevance: Travel Abroad x Apply to Jobs", "https://wwoof.es/en/hosts?map.lat=40.9415270946638&map.lon=-3.71612548828125&map.zoom=7", "Apply to Jobs")
                    ]
    run_unit_test_suite("Low Relevance Tests", testing_pairs)

def borderline_relevance_tests():
    testing_pairs = [("Borderline: AI Research Job x Research AI", "https://job-boards.greenhouse.io/perplexityai/jobs/4537480007", "Research AI"),
                     ("Borderline: AI Research Job x Apply to Jobs", "https://job-boards.greenhouse.io/perplexityai/jobs/4537480007", "Apply to Jobs"),]
    run_unit_test_suite("Borderline Relevance Tests", testing_pairs)

def ambiguous_relevance_tests():
    testing_pairs = [("Google Drive x Research AI", "https://drive.google.com", "Research AI"),
                     ("Google Drive x Apply to Jobs", "https://drive.google.com", "Apply to Jobs"),
                     ("Google Drive x Travel Abroad", "https://drive.google.com", "Travel Abroad"),
                     ("Gmail x Research AI", "https://mail.google.com", "Research AI"),
                     ("Gmail x Apply to Jobs", "https://mail.google.com", "Apply to Jobs"),
                     ("Gmail x Travel Abroad", "https://mail.google.com", "Travel Abroad"),
                     ("Google Calendar x Research AI", "https://calendar.google.com", "Research AI"),
                     ("Google Calendar x Apply to Jobs", "https://calendar.google.com", "Apply to Jobs"),
                     ("Google Calendar x Travel Abroad", "https://calendar.google.com", "Travel Abroad"),]
    run_unit_test_suite("Ambiguous Relevance Tests", testing_pairs)

def run_unit_tests():
    print("Testing relevance scoring with tab content...")
    # high_relevance_tests()
    # low_relevance_tests()
    # borderline_relevance_tests()   
    ambiguous_relevance_tests()     
    print("All tests completed.")

if __name__ == "__main__":
    run_unit_tests()

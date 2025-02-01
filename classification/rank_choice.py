import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

def rank_choice_tab(tab, tasks):
    pass
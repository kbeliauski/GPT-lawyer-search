# Davis Polk Lawyer Query Challenge

## Introduction

Davis Polk is a big law firm with ~1000 lawyers. Here is their website: https://www.davispolk.com/

The goal is to develop a tool that can search for lawyers on their website with extreme specificity.

## Task Description

We want to create a program that can query the lawyers on the Davis Polk website and filter them based on specific criteria. For this challenge, we'll focus on finding lawyers who have worked on cases involving TV networks.

### Examples:
- A lawyer who has worked with a TV network: https://www.davispolk.com/lawyers/sheila-adams-james
- A lawyer who hasn't (for comparison): https://www.davispolk.com/lawyers/faisal-baloch

## Level 1 Challenge

Your task is to write a program that returns a list of all lawyers who have worked on a case with a TV network.

### Expected Output
Your program should produce a list or file containing the names and relevant information of lawyers who meet the criteria.

## Setup and Running the Code

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main program:
   ```
   python main.py
   ```

## Requirements and Constraints

- Use Python for your implementation.
- You may use any libraries listed in the `requirements.txt` file.
- Feel free to use Cursor, ChatGPT, internet, etc.
- Efficiency and latency are extremely important.

## Provided Files

- `README.md`: This file, containing challenge instructions.
- `requirements.txt`: List of required Python packages.
- `main.py`: The main script to run your program.
- `llm_utils.py`: Utility functions for working with language models (if applicable).
- `scraping_utils.py`: Utility functions for web scraping.
- `lawyers.csv`: A CSV file containing initial lawyer data (if provided).

Good luck! 🚀
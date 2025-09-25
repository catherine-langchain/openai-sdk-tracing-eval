# snorkel-ai-evaluation

## Introduction
In the notebook, we will cover how to trace the sample OpenAI SDK agent and run an evaluation using LangSmith. 


## Pre-work

### Clone the repo
```
git clone https://github.com/catherine-langchain/snorkel-ai-evaluation.git
```

## Setup
Follow these instructions to follow along!

### Create an environment and install dependencies  
```
# Ensure you have a recent version of pip and python installed
$ cd snorkel-ai-evaluation
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Running notebooks
Make sure the following command works and opens the relevant notebooks
```
$ jupyter notebook
```

### Set OpenAI API Key
*  Set `OPENAI_API_KEY` in the .env file.

### Sign up for LangSmith and Set LangSmith API Key

*  Sign up [here](https://docs.smith.langchain.com/) 
*  Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, and `LANGSMITH_PROJECT` in the .env file.
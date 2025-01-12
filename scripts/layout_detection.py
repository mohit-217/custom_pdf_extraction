# import os
# import sys
# import os.path as osp
# import argparse

# sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
# import pdf_extract_kit.tasks

# TASK_NAME = 'layout_detection'


# def parse_args():
#     parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
#     parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
#     return parser.parse_args()

# def main(config_path):
#     config = load_config(config_path)
#     task_instances = initialize_tasks_and_models(config)

#     # get input and output path from config
#     input_data = config.get('inputs', None)
#     result_path = config.get('outputs', 'outputs'+'/'+TASK_NAME)

#     # layout_detection_task
#     model_layout_detection = task_instances[TASK_NAME]

#     # for image detection
#     #detection_results = model_layout_detection.predict_images(input_data, result_path)

#     # for pdf detection
#     detection_results = model_layout_detection.predict_pdfs(input_data, result_path)
#     print("upto when")
#     # print(detection_results)
#     print(f'The predicted results can be found at {result_path}')


# if __name__ == "__main__":
#     args = parse_args()
#     main(args.config)
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


class KeyDevelopment(BaseModel):
    """Information about a development in the history of cars."""

    year: int = Field(
        ..., description="The year when there was an important historic development."
    )
    description: str = Field(
        ..., description="What happened in this year? What was the development?"
    )
    evidence: str = Field(
        ...,
        description="Repeat in verbatim the sentence(s) from which the year and description information were extracted",
    )


class ExtractionData(BaseModel):
    """Extracted information about key developments in the history of cars."""

    key_developments: List[KeyDevelopment]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ]
)
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
extractor = prompt | llm.with_structured_output(
    schema=ExtractionData,
    include_raw=False,
)
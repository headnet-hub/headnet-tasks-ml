import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
from core.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskSearchService:
    def __init__(self):
        """
        Initialize the TaskSearchService with the specified model and configuration parameters.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.TASK_SEARCH_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.TASK_SEARCH_MODEL_NAME)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error("Error loading model or tokenizer: %s", e)
            raise e
        
        self.threshold = Config.TASK_SEARCH_THRESHOLD
        self.max_length = Config.TASK_SEARCH_MAX_TOKEN_LENGTH
        self.padding = Config.TASK_SEARCH_PADDING_STRATEGY

    def find_task(self, user_input: str, all_tasks: List[Dict]) -> Optional[str]:
        """
        Find the most relevant task matching the user input from the list of all tasks.

        Args:
        user_input (str): The text input from the user describing the edits.
        all_tasks (list): The list of all tasks. Each task should be a dictionary with the following keys:
            - task_id (string): Unique identifier of the task.
            - task_title (string): Title of the task.
            - list_title (string): Title of the list or category the task belongs to.
            - description (string): Detailed description of the task.
            - deadline (datetime, optional): Deadline of the task in ISO 8601 format.
            - estimation (duration, optional): Estimated time to complete the task (e.g., "3h" for 3 hours).
            - spent (duration, optional): Time spent on the task (e.g., "2h" for 2 hours).
            - priority (string): Priority of the task, can be "none", "low", "medium", "high".

        Returns:
        str: The task_id that best matches the user input, or None if no task matches.
        """
        if not all_tasks:
            logger.warning("No tasks provided for matching.")
            return None

        try:
            # Combine task descriptions with other relevant fields
            task_texts = [f"{task['task_title']} {task['list_title']} {task['description']}" for task in all_tasks]
            inputs = self.tokenizer([user_input] + task_texts, return_tensors='pt', truncation=True, padding=self.padding, max_length=self.max_length)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract the probabilities for the tasks (excluding the user input)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            similarity_scores = probabilities[0, 1:].tolist()

            # Find the index of the most similar task
            most_similar_idx = max(range(len(similarity_scores)), key=similarity_scores.__getitem__)
            if similarity_scores[most_similar_idx] > self.threshold:
                logger.info(f"Task '{all_tasks[most_similar_idx]['task_id']}' exceeds the similarity threshold with a score of {similarity_scores[most_similar_idx]:.4f}.")
                return all_tasks[most_similar_idx]['task_id']
            else:
                logger.info("No task exceeds the similarity threshold.")
                return None

        except Exception as e:
            logger.error("Error during task matching: %s", e)
            return None

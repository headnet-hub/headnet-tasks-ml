import openai
import json
from core.private_config import PrivateConfig
from core.config import Config
from task_search_service import TaskSearchService 

class TaskParserService:
    def __init__(self):
        """
        Initialize the TaskParserService with the OpenAI API key and configuration parameters.
        """
        openai.api_key = PrivateConfig.TASK_PARSER_OPEN_AI_API_KEY
        self.engine = Config.TASK_PARSER_ENGINE
        self.create_prompt_template = Config.TASK_PARSER_CREATE_PROMPT
        self.edit_prompt_template = Config.TASK_PARSER_EDIT_PROMPT
        self.max_tokens = Config.TASK_PARSER_MAX_TOKENS
        self.task_search_service = TaskSearchService()

    def create_task(self, user_input):
        """
        Parse the user input to create a new task.

        Args:
        user_input (str): The text input from the user describing the task.

        Returns:
        dict: A dictionary containing the parsed fields of the task.
        """
        return self._parse_task(user_input, self.create_prompt_template)

    def edit_task(self, user_input, all_tasks):
        """
        Parse the user input to edit an existing task and identify the task to be edited.

        Args:
        user_input (str): The text input from the user describing the edits.
        all_tasks (list): The list of all tasks.

        Returns:
        dict: A dictionary containing the parsed fields of the task and the task ID.
        """
        previous_task_info = self._find_task_info(user_input, all_tasks)
        if previous_task_info:
            prompt = self.edit_prompt_template.format(previous_task_info=previous_task_info, text=user_input)
            parsed_task = self._parse_task(prompt, self.edit_prompt_template)
            if parsed_task:
                parsed_task['id'] = previous_task_info['task_id']
                return parsed_task
        return None

    def _parse_task(self, text, prompt_template):
        """
        Use the OpenAI API to parse the task description.

        Args:
        text (str): The text description of the task.
        prompt_template (str): The template to format the prompt.

        Returns:
        dict: A dictionary containing the parsed fields of the task.
        """
        prompt = prompt_template.format(text=text)
        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                max_tokens=self.max_tokens
            )
            return self._parse_response(response)
        except openai.error.OpenAIError as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    def _parse_response(self, response):
        """
        Parse the response from the OpenAI API to extract task fields.

        Args:
        response (openai.Completion): The response object from the OpenAI API.

        Returns:
        dict: A dictionary containing the parsed fields of the task.
        """
        try:
            task_data = json.loads(response.choices[0].text.strip())
            return {
                "title": task_data.get("title"),
                "description": task_data.get("description"),
                "deadline": task_data.get("deadline"),
                "priority": task_data.get("priority"),
                "estimated_time": task_data.get("estimated_time"),
                "spent_time": task_data.get("spent_time")
            }
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return None

    def _find_task_info(self, user_input, all_tasks):
        """
        Use TaskSearchService to find the most relevant task information based on user input.

        Args:
        user_input (str): The text input from the user describing the edits.
        all_tasks (list): The list of all tasks.

        Returns:
        dict: A dictionary containing the task information.
        """
        task_id = self.task_search_service.find_task(user_input, all_tasks)
        if task_id:
            for task in all_tasks:
                if task['task_id'] == task_id:
                    return task
        return None

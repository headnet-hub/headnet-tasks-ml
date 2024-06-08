from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import Config
from core.private_config import PrivateConfig

class CategorizerService:
    def __init__(self):
        """
        Initialize the CategorizerService with the model name, authorization token, and classification threshold.
        """
        self.model_name = Config.CATEGORIZER_MODEL_NAME
        self.auth_token = PrivateConfig.CATEGORIZER_AUTH_TOKEN
        self.threshold_classification = Config.CATEGORIZER_TRESHOLD_CLASSIFICATION
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """
        Load the tokenizer and model based on the initialized model name and authorization token.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.auth_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, use_auth_token=self.auth_token)
        
    def categorize(self, text, categories):
        """
        Categorize the given text into the provided categories.

        Args:
        text (str): The text to be categorized.
        categories (list of str): A list of category names to which the text could belong.

        Returns:
        list of str: List of categories that are applicable to the text based on the classification threshold.
        """
        if not self.tokenizer or not self.model:
            self.load_model_and_tokenizer()

        results = []
        for category in categories:
            hypothesis = f"Это пример {category}."
            inputs = self.tokenizer.encode(text, hypothesis, return_tensors='pt', truncation=True)
            outputs = self.model(inputs)
            logits = outputs.logits
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1].item()
            if prob_label_is_true > self.threshold_classification:
                results.append(category)
        return results

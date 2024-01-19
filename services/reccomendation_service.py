import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from core.config import Config

class RecommendationService:
    def __init__(self):
        """
        Initialize the RecommendationService with the configuration parameters.
        """
        self.min_scale = Config.MIN_SCALE
        self.max_scale = Config.MAX_SCALE
        self.impact_score_weight = Config.IMPACT_SCORE_WEIGHT
        self.preference_bonus_weight = Config.PREFERENCE_BONUS_WEIGHT
        self.completed_count_weight = Config.COMPLETED_COUNT_WEIGHT
        self.agreed_count_weight = Config.AGREED_COUNT_WEIGHT
        self.showed_count_weight = Config.SHOWED_COUNT_WEIGHT
        self.test_size = Config.RECCOMENDATIONS_TEST_SIZE
    
    def _scale_score(self, score):
        """
        Scale the given score to be within the min and max scale.

        Args:
        score (float): The score to be scaled.

        Returns:
        float: The scaled score.
        """
        return max(self.min_scale, min(self.max_scale, round(score)))
    
    def _calculate_advice_rating(self, advice, user_likes_categories, user_category_impact):
        """
        Calculate the advice rating based on user preferences and advice interactions.

        Args:
        advice (object): The advice object.
        user_likes_categories (list of str): List of categories the user likes.
        user_category_impact (dict): A dictionary mapping category names to their impact on the user.

        Returns:
        float: The calculated advice rating.
        """
        impact_score = sum(user_category_impact.get(cat, 0) for cat in advice.categories)
        user_preference_bonus = sum(1 for cat in advice.categories if cat in user_likes_categories)
        interaction_score = self.agreed_count_weight * advice.agreed_count + self.completed_count_weight * advice.completed_count + self.showed_count_weight * advice.showed_count
        total_score = self.impact_score_weight * impact_score + self.preference_bonus_weight * user_preference_bonus + interaction_score
        scaled_score = self._scale_score(total_score)
        return scaled_score
    
    def _prepare_surprise_data(self, user_categories, user_category_impact, advice_objects):
        """
        Prepare data for training the Surprise model.

        Args:
        user_categories (list of str): List of categories the user likes.
        user_category_impact (dict): A dictionary mapping category names to their impact on the user.
        advice_objects (dict): A dictionary of advice objects.

        Returns:
        DataFrame: The prepared data as a pandas DataFrame.
        """
        records = []
        for advice_name, advice in advice_objects.items():
            records.append(('user', advice_name, self._calculate_advice_rating(advice, user_categories, user_category_impact)))
        return pd.DataFrame(records, columns=['userID', 'itemID', 'rating'])
    
    def _create_surprise_dataset(self, dataframe):
        """
        Create a Surprise dataset from the pandas DataFrame.

        Args:
        dataframe (DataFrame): The pandas DataFrame containing the data.

        Returns:
        Dataset: The prepared Surprise dataset.
        """
        reader = Reader(rating_scale=(self.min_scale, self.max_scale))
        return Dataset.load_from_df(dataframe[['userID', 'itemID', 'rating']], reader)
    
    def _train_and_evaluate_surprise_model(self, dataset):
        """
        Train and evaluate the Surprise model.

        Args:
        dataset (Dataset): The Surprise dataset.

        Returns:
        SVD: The trained SVD model.
        """
        trainset, testset = train_test_split(dataset, test_size=self.test_size)
        model = SVD()
        model.fit(trainset)
        predictions = model.test(testset)
        accuracy.rmse(predictions)
        return model
    
    def _get_surprise_recommendations(self, model, advice_objects, user_id='user'):
        """
        Get recommendations for the user.

        Args:
        model (SVD): The trained SVD model.
        advice_objects (dict): A dictionary of advice objects.
        user_id (str): The user's ID.

        Returns:
        str: The name of the best advice.
        """
        recommendations = []
        for advice_name in advice_objects.keys():
            prediction = model.predict(user_id, advice_name)
            recommendations.append((prediction.iid, prediction.est))
        best_advice = max(recommendations, key=lambda x: x[1])[0] 
        return best_advice
    
    def get_recommendations(self, user_categories, user_category_impact, advice_objects):
        """
        Get recommendations based on user preferences and past interactions with advice.

        Args:
        user_categories (list of str): List of categories the user likes.
        user_category_impact (dict): A dictionary mapping category names to their impact on the user.
        advice_objects (dict): A dictionary of advice objects.

        Returns:
        str: The name of the best advice.
        """
        surprise_data = self._prepare_surprise_data(user_categories, user_category_impact, advice_objects)
        surprise_dataset = self._create_surprise_dataset(surprise_data)
        model = self._train_and_evaluate_surprise_model(surprise_dataset)
        return self._get_surprise_recommendations(model, advice_objects)

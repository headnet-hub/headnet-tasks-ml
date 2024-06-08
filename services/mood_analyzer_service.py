import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from core.config import Config

class MoodAnalyzerService:
    def __init__(self):
        """
        Initialize the MoodAnalyzerService with the parameters for the RandomForestRegressor and analysis settings.
        """
        self.n_estimators = Config.MOOD_ANALYZER_N_ESTIMATORS
        self.random_state = Config.MOOD_ANALYZER_RANDOM_STATE
        self.test_size = Config.MOOD_ANALYZER_TEST_SIZE
        self.threshold_importance = Config.MOOD_ANALYZER_TRESHOLD_IMPORTANCE
        self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        
    def _prepare_data(self, df):
        """
        Prepare data for training and analysis.

        Args:
        df (DataFrame): The dataframe containing the user history data.

        Returns:
        tuple: Feature matrix X and target variable y.
        """
        y = df['mood_rate']
        X = df.drop(['mood_rate'], axis=1)
        return X, y

    def _train(self, X, y):
        """
        Train the RandomForestRegressor model on the provided data.

        Args:
        X (DataFrame): The feature matrix.
        y (Series): The target variable.

        Returns:
        float: The R^2 score of the model on the test set.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.rf_model.fit(X_train, y_train)
        return self.rf_model.score(X_test, y_test)
    
    def analyze_mood(self, user_history):
        """
        Analyze mood based on user history.

        Args:
        user_history (dict): The user's historical data with features and mood ratings.

        Returns:
        dict: A dictionary mapping feature names to their calculated impact on mood.
        """
        df = pd.DataFrame(user_history)
        if 'mood_rate' not in df.columns:
            raise ValueError("The data must include a 'mood_rate' column.")
        X, y = self._prepare_data(df)
        
        self._train(X, y)
       
        feature_importances = self.rf_model.feature_importances_
        feature_names = X.columns
        
        importances = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
        correlation_matrix = df.corr()
        correlation_with_mood = correlation_matrix['mood_rate'].sort_values(ascending=False)
        
        importances_df = importances.reset_index()
        importances_df.columns = ['Feature', 'Importance']

        correlation_with_mood_df = correlation_with_mood.reset_index()
        correlation_with_mood_df.columns = ['Feature', 'Correlation']

        combined_df = importances_df.merge(correlation_with_mood_df, on='Feature')
        combined_df['Impact'] = combined_df['Importance'] * combined_df['Correlation']

        important_features_df = combined_df[combined_df['Importance'] >= self.threshold_importance]
        important_features_df = important_features_df.sort_values(by='Impact', ascending=False)
        important_features_df = important_features_df.reset_index(drop=True)

        feature_impact_map = important_features_df.set_index('Feature')['Impact'].to_dict()

        return feature_impact_map

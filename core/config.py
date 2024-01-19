class Config:
    # CategorizerService Configurations
    MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
    AUTH_TOKEN = "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"
    TRESHOLD_CLASSIFICATION = 0.95
    
    # MoodAnalyzerService Configurations
    N_ESTIMATORS = 100
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TRESHOLD_IMPORTANCE = 0.05
    
    # RecommendationService Configurations
    MIN_SCALE = 1
    MAX_SCALE = 5
    IMPACT_SCORE_WEIGHT = 250
    PREFERENCE_BONUS_WEIGHT = 50
    COMPLETED_COUNT_WEIGHT = 3
    AGREED_COUNT_WEIGHT = 2
    SHOWED_COUNT_WEIGHT = -1
    RECCOMENDATIONS_TEST_SIZE = 0.25
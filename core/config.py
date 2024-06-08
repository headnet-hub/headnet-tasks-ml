class Config:
    # CategorizerService Configurations
    CATEGORIZER_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
    CATEGORIZER_TRESHOLD_CLASSIFICATION = 0.95
    
    # MoodAnalyzerService Configurations
    MOOD_ANALYZER_N_ESTIMATORS = 100
    MOOD_ANALYZER_RANDOM_STATE = 42
    MOOD_ANALYZER_TEST_SIZE = 0.2
    MOOD_ANALYZER_TRESHOLD_IMPORTANCE = 0.05
    
    # RecommendationService Configurations
    RECOMMENDATION_MIN_SCALE = 1
    RECOMMENDATION_MAX_SCALE = 5
    RECOMMENDATION_IMPACT_SCORE_WEIGHT = 250
    RECOMMENDATION_PREFERENCE_BONUS_WEIGHT = 50
    RECOMMENDATION_COMPLETED_COUNT_WEIGHT = 3
    RECOMMENDATION_AGREED_COUNT_WEIGHT = 2
    RECOMMENDATION_SHOWED_COUNT_WEIGHT = -1
    RECCOMENDATION_TEST_SIZE = 0.25
    
    # TaskSearchService Configurations
    TASK_SEARCH_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
    TASK_SEARCH_THRESHOLD = 0.9
    TASK_SEARCH_MAX_TOKEN_LENGTH = 512
    TASK_SEARCH_PADDING_STRATEGY = 'max_length'
    
    # TaskParserService Configurations
    TASK_PARSER_ENGINE = "gpt-4o"
    TASK_PARSER_CREATE_PROMPT = (
        "Parse the following task: {text}\n\n"
        "Please provide the title, description, deadline (YYYY-MM-DDTHH:MM), priority (none, low, medium, high), "
        "estimated_time (in minutes), and spent_time (in minutes) in a JSON format."
    )
    TASK_PARSER_EDIT_PROMPT = (
        "You are editing an existing task with the following details:\n"
        "{previous_task_info}\n\n"
        "The user provided the following updates:\n{text}\n\n"
        "Please provide the updated title, description, deadline (YYYY-MM-DDTHH:MM), priority (none, low, medium, high), "
        "estimated_time (in minutes), and spent_time (in minutes) in a JSON format."
    )
    TASK_PARSER_MAX_TOKENS = 150
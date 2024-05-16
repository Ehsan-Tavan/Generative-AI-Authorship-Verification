"""
    Generative_AI_Authorship_Verification Project:
        Make the importing much shorter
"""
from .data_preparation import create_samples, sequence_classification_data_creator, \
    create_single_samples, paraphraser_data_creator, generation_data_creator, \
    sequence_classification_data_creator2, instruction_tuning_data_creator
from .training_data_preparation import prepare_data, create_paraphraser_data

import config


class ActiveConfig:

    # fixmeX
    def __init__(self, user_args) -> None:
        self.path_to_spectrogram = config.path_to_spectrogram.format(user_args.run_id)
        self.path_to_test_spectrogram = config.path_to_test_spectrogram.format(user_args.run_id)
        self.path_to_slices_for_training = config.path_to_slices_for_training.format(user_args.run_id)
        self.path_to_slices_for_testing = config.path_to_slices_for_testing.format(user_args.run_id)

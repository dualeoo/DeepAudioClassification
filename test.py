import utility


class Test:
    def __init__(self, user_args, dataset, my_logger, model, path_to_model) -> None:
        self.user_args = user_args
        self.dataset = dataset
        self.my_logger = my_logger
        self.model = model
        self.path_to_model = path_to_model
        self.load_model()

    def predict(self):
        time_starting = utility.log_time_start(self.user_args.mode)

        x_np = self.dataset.x_np
        file_names = self.dataset.file_names

        final_result = {}
        predict_result = self.model.predict(x_np)

        self.my_logger.warning("Stop predicting because starting from this point forward coudd be wrong. "
                               "Please start debugging!")
        exit()

        # TODO be careful. Starting from here could be wrong
        predict_result = utility.preprocess_predict_result(predict_result)  # TODOx look inside
        # TODO change the name of method save_predict_result
        utility.save_predict_result(predict_result, file_names, final_result)  # TODOx look inside
        final_result = utility.finalize_result(final_result)  # TODOx look inside
        utility.save_final_result(final_result)  # TODOx look inside

        utility.log_time_end(self.user_args.mode, time_starting)

    def load_model(self):
        self.my_logger.info("[+] Loading weights...")
        self.model.load(self.path_to_model)
        self.my_logger.info("[+] Weights loaded! ✅")

    def evaluate(self):
        time_starting = utility.log_time_start(self.user_args.mode)

        test_X = self.dataset.x_np
        test_y = self.dataset.y_np

        testAccuracy = self.model.evaluate(test_X, test_y)[0]
        self.my_logger.info("[+] Test accuracy: {} ".format(testAccuracy))

        utility.log_time_end(self.user_args.mode, time_starting)

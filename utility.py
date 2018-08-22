def process_file_name(file_name):
    split_result = file_name.split("_")
    return split_result[2], split_result[3]  # TODOx


def save_predict_result(predict_results, file_names, final_result):
    for i in range(len(predict_results)):
        predict_result = predict_results[i]
        file_name = file_names[i]
        file_name, slice_id = process_file_name(file_name)
        if file_name not in final_result:
            final_result[file_name] = {}
        result = final_result[file_name]
        if predict_result not in result:
            result[predict_result] = 1
        else:
            result[predict_result] += 1
    return final_result


def preprocess_predict_result(predict_results):
    new_result = []
    for result in predict_results:
        new_result.append(result[0])
    return new_result  # TODOx

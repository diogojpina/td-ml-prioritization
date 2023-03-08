def accuracy_tuned(target_expected, target_predicted):
    hit = 0
    for i in range(len(target_expected)):
        diff = abs(target_expected[i] - target_predicted[i])
        penalty = diff / 5
        hit = hit + (1 - penalty)

    accuracy = hit / len(target_expected)
    return accuracy

def precision_tuned(target_expected, target_predicted, labels):
    class_labels_size = len(labels)
    hit_sum = 0
    for labelIdx in range(len(labels)):
        class_label = labels[labelIdx]
        # print('Class: ', class_label)

        hit_count = 0
        hit = 0
        for i in range(len(target_expected)):
            if (target_predicted[i] == class_label):
                # print(target_predicted[i])
                diff = abs(target_expected[i] - target_predicted[i])
                penalty = diff / (class_labels_size - 1)
                hit = hit + (1 - penalty)

                # if (target_expected[i] == target_predicted[i]):
                #     hit = hit + 1

                hit_count = hit_count + 1

        if (hit_count > 0):
            precision = hit / hit_count
            hit_sum = hit_sum + precision
        
        # print(hit)
        # print(hit_count)
        # print('Precision: ', precision)

    # print(hit_sum)
    # print(len(labels))
    precision_mean = hit_sum / len(labels)
    # print(precision)
    return precision_mean

def recall_tuned(target_expected, target_predicted, labels):
    class_labels_size = len(labels)
    hit_sum = 0
    for labelIdx in range(len(labels)):
        class_label = labels[labelIdx]
        # print('Class: ', class_label)

        hit_count = 0
        hit = 0
        for i in range(len(target_expected)):
            if (target_expected[i] == class_label):
                # print(target_test[i])
                diff = abs(target_expected[i] - target_predicted[i])
                penalty = diff / (class_labels_size - 1)
                hit = hit + (1 - penalty)

                # if (target_expected[i] == target_predicted[i]):
                #     hit = hit + 1

                hit_count = hit_count + 1

        if (hit_count > 0):
            recall = hit / hit_count
            hit_sum = hit_sum + recall
        
        # print(hit)
        # print(hit_count)
        # print('Precision: ', precision)

    # print(hit_sum)
    # print(len(labels))
    recall_mean = hit_sum / len(labels)
    # print(precision)
    return recall_mean

def f1_tuned(target_expected, target_predicted, labels):
    precision = precision_tuned(target_expected, target_predicted, labels)
    recall = recall_tuned(target_expected, target_predicted, labels)

    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    
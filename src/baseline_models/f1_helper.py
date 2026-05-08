def harmonic_mean(a,b):
    return 2/(1/a+1/b)

def accuracy(y_arr, p_arr):
    n = len(y_arr)

    tn = sum((1-y_arr[i]) * (1-p_arr[i]) for i in range(n))
    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fn = sum(y_arr[i] * (1-p_arr[i]) for i in range(n))
    fp = sum((1-y_arr[i]) * p_arr[i] for i in range(n))

    return (tn+tp)/(fn+fp+tn+tp)

def precision(y_arr, p_arr):
    n = len(y_arr)

    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fp = sum((1-y_arr[i]) * p_arr[i] for i in range(n))

    return tp/(tp+fp)

def recall(y_arr, p_arr):
    n = len(y_arr)

    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fn = sum(y_arr[i] * (1-p_arr[i]) for i in range(n))

    return tp/(tp+fn)

def f1(y_arr, p_arr):
    n = len(y_arr)

    tp = sum(y_arr[i] * p_arr[i] for i in range(n))
    fn = sum(y_arr[i] * (1-p_arr[i]) for i in range(n))
    fp = sum((1-y_arr[i]) * p_arr[i] for i in range(n))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return harmonic_mean(precision, recall)

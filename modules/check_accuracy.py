# calculate metrics for accuracy of machine learning algorithms
# Matthew J. Neave 02.08.16

def check_accuracy(answers, predicted):

    from sklearn import metrics

    # simple accuracy score
    print "accuracy_score:", metrics.accuracy_score(answers[0::,1], predicted)

    # F1 score
    print "F1_score:", metrics.f1_score(answers[0::,1], predicted, average="weighted", pos_label=None)


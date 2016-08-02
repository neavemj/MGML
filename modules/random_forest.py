# train random forest classifier using gc content and kmer frequencies
# Matthew J. Neave

def random_forest(train_array, test_array, n_est):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # create random forest object
    forest = RandomForestClassifier(n_estimators=n_est)

    # fit training data
    forest = forest.fit(train_array[0::,2::],train_array[0::,1])

    # predict accuracy against test data
    output = forest.predict(test_array[0::,2::])

    return output

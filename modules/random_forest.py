# train random forest classifier using gc content and kmer frequencies
# Matthew J. Neave

def random_forest(train_array, test_array, n_est, max_depth, threads):

    from sklearn.ensemble import RandomForestClassifier

    # create random forest object
    forest = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, \
            verbose=1, n_jobs=threads)

    # fit training data
    forest = forest.fit(train_array[0::,2::],train_array[0::,1])

    # predict accuracy against test data
    output = forest.predict(test_array[0::,2::])

    return forest, output

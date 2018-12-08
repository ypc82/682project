

def countRelationDifference(data_type = 'WEBQSP'):
    """
    :param data_type:
    :return: Number of unique relations, number of relations that only appear in test set but not training set
    """

    if data_type == 'SQ':
        train_data_path = 'data/sq_relations/train.replace_ne.withpool'
        test_data_path = 'data/sq_relations/test.replace_ne.withpool'
    else:
        train_data_path = 'data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt'
        test_data_path = 'data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt'

    training_relation = set()
    test_relation = set()

    with open(train_data_path) as infile:
        for idx, line in enumerate(infile, 1):
            for s in line.split():
                if s.isdigit():
                    training_relation.add(int(s))

    with open(test_data_path) as infile:
        for idx, line in enumerate(infile, 1):
            for s in line.split():
                if s.isdigit():
                    test_relation.add(int(s))

    # Relations in test set but not training set
    relation_set_difference = test_relation.difference(training_relation)

    print("Number of relation in training set: ", len(training_relation))
    print("Number of relation in test set: ", len(test_relation))
    print("Relation in test set but not training set", len(relation_set_difference))


countRelationDifference()



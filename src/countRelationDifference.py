import math

def load_relations_map(path):
    ''' Return self.relations_map = {idx:relation_names}
    '''
    relations_map = {}
    print('Load', path)
    with open(path) as infile:
        for idx, line in enumerate(infile, 1):
            relations_map[idx] = line.split("..")
    return relations_map

def getRelationStats(path, relation_map):

    relation = set()
    total_n_question = 0
    avg_n, max_n, min_n  = 0, 0, math.inf

    with open(path) as infile:
        for idx, line in enumerate(infile, 1):
            count = 0
            total_n_question += 1

            for s in line.split():
                if s.isdigit():
                    count += len(relation_map[int(s)])
                    avg_n += len(relation_map[int(s)])
                    relation.add((int(s), len(relation_map[int(s)])))
                else:
                    break

            if count > max_n:
                max_n = count

            if count < min_n:
                min_n = count

        avg_n /= total_n_question
    return relation, max_n, min_n, avg_n

def countRelation(relation_set):
    """
    :param relation_set: a set that cotains tuples (relation_index, num_of_relation)
    :return: number of relations in the set
    """
    num_of_relation = 0
    for relation_tuple in relation_set:

        num_of_relation += relation_tuple[1]

    return num_of_relation

def countRelationDifference(data_type = 'WEBQSP'):
    """
    :param data_type:
    :return: Number of unique relations, number of relations that only appear in test set but not training set
    """

    if data_type == 'SQ':
        train_data_path = 'data/sq_relations/train.replace_ne.withpool'
        test_data_path = 'data/sq_relations/test.replace_ne.withpool'
        relation_map = load_relations_map('data/sq_relations/relation.2M.list')
    else:
        train_data_path = 'data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt'
        test_data_path = 'data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt'
        relation_map = load_relations_map('data/webqsp_relations/relations.txt')

    training_relation, max_n_train, min_n_train, avg_n_train = getRelationStats(train_data_path, relation_map)
    test_relation, max_n_test, min_n_test, avg_n_test = getRelationStats(test_data_path, relation_map)

    # Relations in test set but not training set
    relation_set_difference = test_relation.difference(training_relation)

    relation_num_training = countRelation(training_relation)
    relation_num_test = countRelation(test_relation)
    relation_num_difference = countRelation(relation_set_difference)

    print("Number of relation in training set: ", relation_num_training)
    print("average relation number in training set: ", avg_n_train)
    print("max relation number in a training set question: ", max_n_train)
    print("min relation number in a training set question: ", min_n_train)

    print("Number of relation in test set: ", relation_num_test)
    print("average relation number in test set: ", avg_n_test)
    print("max relation number in a test set question: ", max_n_test)
    print("min relation number in a test set question: ", min_n_test)

    print("Relation in test set but not training set", relation_num_difference)



countRelationDifference("WQ")



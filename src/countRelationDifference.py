def load_relations_map(path):
    ''' Return self.relations_map = {idx:relation_names}
    '''
    relations_map = {}
    print('Load', path)
    with open(path) as infile:
        for idx, line in enumerate(infile, 1):
            relations_map[idx] = line.split("..")
    return relations_map

def getRelation(path, relation_map):

    relation = set()

    with open(path) as infile:
        for idx, line in enumerate(infile, 1):
            for s in line.split():
                if s.isdigit():
                    relation.add((int(s), len(relation_map[int(s)])))
                else:
                    break
    return relation

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

    training_relation = getRelation(train_data_path, relation_map)
    test_relation = getRelation(test_data_path, relation_map)

    # Relations in test set but not training set
    relation_set_difference = test_relation.difference(training_relation)

    relation_num_training = countRelation(training_relation)
    relation_num_test = countRelation(test_relation)
    relation_num_difference = countRelation(relation_set_difference)

    print("Number of relation in training set: ", relation_num_training)
    print("Number of relation in test set: ", relation_num_test)
    print("Relation in test set but not training set", relation_num_difference)


countRelationDifference()



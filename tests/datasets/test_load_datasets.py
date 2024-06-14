def test_load_dataset():
    from holisticai.datasets import load_dataset
    SHARD_SIZE = 50
    dataset = load_dataset("adult")
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['sex']=="Male", 'group_b': x['p_attr']['sex']=="Female"}, vectorized=True)
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE)
    dataset = dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)    

def test2():
    from holisticai.datasets import load_dataset
    dataset = load_dataset('law_school', protected_attribute='race1')
    dataset = dataset.map(lambda x: {'group_c':x['group_a'], 'group_w':x['group_b']}, vectorized=True)

test_load_dataset()
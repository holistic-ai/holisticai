def test_load_dataset():
    from holisticai.datasets import load_dataset
    SHARD_SIZE = 50
    dataset = load_dataset("adult", protected_attribute='sex').sample(1000)
    dataset = dataset.map(lambda x: {'new': str(x['group_a'])  + str(x['y'])}, vectorized=False)
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE)
    dataset = dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)
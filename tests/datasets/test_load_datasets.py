def test_load_dataset():
    from holisticai.datasets import load_dataset
    SHARD_SIZE = 50
    dataset = load_dataset("adult")
    dataset = dataset.rename({"x":"X"})
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']}, vectorized=True)
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE)
    dataset = dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)    
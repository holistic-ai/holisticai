import numpy as np


class ActiveLearningItemsSelection:
    def __init__(self, top_n=10, theta=3):
        self.top_n = top_n
        self.theta = theta

    def __call__(self, time_mask, pred_data, return_only_time_mask=False):
        pred_data = 5 - np.abs(self.theta - pred_data)
        candidate_index = ~time_mask  ## convert all candidate index to True
        candidate_rating = pred_data * candidate_index  ## get predicte rating on those candidate index
        sorted_ind = np.argsort(-candidate_rating, axis=1)[
            :, :
        ]  ## get the index of each row, where top n pred are selected
        items_selected_matrix = []
        for row, selected_index in enumerate(sorted_ind):
            candidate_index[row, selected_index] = False
            items_selected_matrix.append(selected_index)
        items_selected_matrix = np.array(items_selected_matrix)
        time_mask = ~candidate_index  ## get the new index
        if return_only_time_mask:
            return time_mask
        return items_selected_matrix, time_mask


class ConventionalItemsSelection:
    def __init__(self, top_n=10):
        self.top_n = top_n

    def __call__(self, time_mask, pred_data, return_only_time_mask=False):
        candidate_index = ~time_mask  ## convert all candidate index to True
        candidate_rating = pred_data * candidate_index  ## get predicte rating on those candidate index
        sorted_ind = np.argsort(-candidate_rating, axis=1)[
            :, : self.top_n
        ]  ## get the index of each row, where top n pred are selected
        items_selected_matrix = []
        for row, selected_index in enumerate(sorted_ind):
            candidate_index[row, selected_index] = False
            items_selected_matrix.append(selected_index)
        items_selected_matrix = np.array(items_selected_matrix)
        time_mask = ~candidate_index  ## get the new index
        if return_only_time_mask:
            return time_mask
        return items_selected_matrix, time_mask


class RandomItemSelection:
    def __init__(self, top_n=10):
        self.top_n = top_n

    def __call__(self, time_mask, pred_data, return_only_time_mask=False):
        candidate_index = ~time_mask  ## convert all candidate index to True
        candidate_rating = pred_data * candidate_index  ## get predicte rating on those candidate index
        sorted_ind = np.argsort(-candidate_rating, axis=1)[
            :, :
        ]  ## get the index of each row, where top n pred are selected
        items_selected_matrix = []
        for row, selected_index in enumerate(sorted_ind):
            item_selected = np.random.choice(selected_index, self.top_n, replace=False)
            candidate_index[row, item_selected] = False
            items_selected_matrix.append(item_selected)
        items_selected_matrix = np.array(items_selected_matrix)
        time_mask = ~candidate_index  ## get the new index
        if return_only_time_mask:
            return time_mask
        return items_selected_matrix, time_mask

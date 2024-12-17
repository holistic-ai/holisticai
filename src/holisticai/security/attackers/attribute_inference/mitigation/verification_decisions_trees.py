# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements robustness verifications for decision-tree-based models.
"""
# from __future__ import absolute_import, division, print_function, unicode_literals

from __future__ import annotations

from typing import Optional


class Interval:
    """
    Representation of an intervals bound.
    """

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        """
        An interval of a feature.

        :param lower_bound: The lower boundary of the feature.
        :param upper_bound: The upper boundary of the feature.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class Box:
    """
    Representation of a box of intervals bounds.
    """

    def __init__(self, intervals: Optional[dict[int, Interval]] = None) -> None:
        """
        A box of intervals.

        :param intervals: A dictionary of intervals with features as keys.
        """
        if intervals is None:
            self.intervals = {}
        else:
            self.intervals = intervals

    def intersect_with_box(self, box: Box) -> None:
        """
        Get the intersection of two interval boxes. This function modifies this box instance.

        :param box: Interval box to intersect with this box.
        """
        for key, value in box.intervals.items():
            if key not in self.intervals:
                self.intervals[key] = value
            else:
                lower_bound = max(self.intervals[key].lower_bound, value.lower_bound)
                upper_bound = min(self.intervals[key].upper_bound, value.upper_bound)

                if lower_bound >= upper_bound:  # pragma: no cover
                    self.intervals.clear()
                    break

                self.intervals[key] = Interval(lower_bound, upper_bound)

    def get_intersection(self, box: Box) -> Box:
        """
        Get the intersection of two interval boxes. This function creates a new box instance.

        :param box: Interval box to intersect with this box.
        """
        box_new = Box(intervals=self.intervals.copy())

        for key, value in box.intervals.items():
            if key not in box_new.intervals:
                box_new.intervals[key] = value
            else:
                lower_bound = max(box_new.intervals[key].lower_bound, value.lower_bound)
                upper_bound = min(box_new.intervals[key].upper_bound, value.upper_bound)

                if lower_bound >= upper_bound:
                    box_new.intervals.clear()
                    return box_new

                box_new.intervals[key] = Interval(lower_bound, upper_bound)

        return box_new

    def __repr__(self):
        return self.__class__.__name__ + f"({self.intervals})"


class LeafNode:
    """
    Representation of a leaf node of a decision tree.
    """

    def __init__(
        self,
        tree_id: Optional[int],
        class_label: int,
        node_id: Optional[int],
        box: Box,
        value: float,
    ) -> None:
        """
        Create a leaf node representation.

        :param tree_id: ID of the decision tree.
        :param class_label: ID of class to which this leaf node is contributing.
        :param box: A box representing the n_feature-dimensional bounding intervals that reach this leaf node.
        :param value: Prediction value at this leaf node.
        """
        self.tree_id = tree_id
        self.class_label = class_label
        self.node_id = node_id
        self.box = box
        self.value = value

    def __repr__(self):
        return (
            self.__class__.__name__ + f"({self.tree_id}, {self.class_label}, {self.node_id}, {self.box}, {self.value})"
        )


class Tree:
    """
    Representation of a decision tree.
    """

    def __init__(self, class_id: Optional[int], leaf_nodes: list[LeafNode]) -> None:
        """
        Create a decision tree representation.

        :param class_id: ID of the class to which this decision tree contributes.
        :param leaf_nodes: A list of leaf nodes of this decision tree.
        """
        self.class_id = class_id
        self.leaf_nodes = leaf_nodes

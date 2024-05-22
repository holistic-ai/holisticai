from scipy.stats import binom

from . import mtable_generator

EPS = 0.0000000000000001


class RecursiveNumericFailProbabilityCalculator:
    """Recursive calculation of fail probability"""

    def __init__(self, k, p, alpha):
        self.k = k
        self.p = p
        self.alpha = alpha
        self.pmf_cache = {}
        self.legal_assignment_cache = {}

    def adjust_alpha(self):
        a_min = 0
        a_max = self.alpha
        a_mid = (a_min + a_max) / 2

        minb = self._compute_boundary(a_min)
        maxb = self._compute_boundary(a_max)
        midb = self._compute_boundary(a_mid)

        while (
            minb.mass_of_mtable() < maxb.mass_of_mtable()
            and midb.fail_prob != self.alpha
        ):
            if midb.fail_prob < self.alpha:
                a_min = a_mid
                minb = self._compute_boundary(a_min)
            elif midb.fail_prob > self.alpha:
                a_max = a_mid
                maxb = self._compute_boundary(a_max)

            a_mid = (a_min + a_max) / 2
            midb = self._compute_boundary(a_mid)

            max_mass = maxb.mass_of_mtable()
            min_mass = minb.mass_of_mtable()
            mid_mass = midb.mass_of_mtable()

            if max_mass - min_mass == 1 or maxb.alpha - minb.alpha <= EPS:
                min_diff = abs(minb.fail_prob - self.alpha)
                max_diff = abs(maxb.fail_prob - self.alpha)

                if min_diff <= max_diff:
                    return minb
                else:
                    return maxb

            if max_mass - mid_mass == 1 and mid_mass - min_mass == 1:
                min_diff = abs(minb.fail_prob - self.alpha)
                max_diff = abs(maxb.fail_prob - self.alpha)
                mid_diff = abs(midb.fail_prob - self.alpha)

                if mid_diff <= max_diff and mid_diff <= min_diff:
                    return midb
                if min_diff <= mid_diff and min_diff <= max_diff:
                    return minb
                else:
                    return maxb

        return midb

    def calculate_fail_probability(self, mtable):
        """
        Analytically calculates the fail probability of the mtable
        """
        aux_mtable = mtable_generator.compute_aux_mtable(mtable)
        max_protected = aux_mtable["block"].sum()
        block_sizes = aux_mtable["block"].tolist()  # [1:]
        success_prob = self._find_legal_assignments(max_protected, block_sizes)
        return 0 if success_prob == 0 else (1 - success_prob)

    def get_from_pmf_cache(self, trials, successes):
        key = (trials, successes)
        if not key in self.pmf_cache:
            self.pmf_cache[key] = binom.pmf(k=successes, n=trials, p=self.p)
        return self.pmf_cache[key]

    def _compute_boundary(self, alpha):
        """
        Returns a tuple of (k, p, alpha, fail_prob, mtable)
        """
        mtable = mtable_generator.MTableGenerator(
            self.k, self.p, alpha
        ).mtable_as_dataframe()
        fail_prob = self.calculate_fail_probability(mtable)
        return MTableFailProbPair(self.k, self.p, alpha, fail_prob, mtable)

    def _find_legal_assignments(self, number_of_candidates, block_sizes):
        return self._find_legal_assignments_aux(number_of_candidates, block_sizes, 1, 0)

    def _find_legal_assignments_aux(
        self,
        number_of_candidates,
        block_sizes,
        current_block_number,
        candidates_assigned_so_far,
    ):
        if len(block_sizes) == 0:
            return 1

        min_needed_this_block = current_block_number - candidates_assigned_so_far
        if min_needed_this_block < 0:
            min_needed_this_block = 0

        max_possible_this_block = min(block_sizes[0], number_of_candidates)

        assignments = 0
        new_remaining_block_sizes = block_sizes[1:]
        for items_this_block in range(
            min_needed_this_block, max_possible_this_block + 1
        ):
            new_remaining_candidates = number_of_candidates - items_this_block

            suffixes = self._calculate_legal_assignments_aux(
                new_remaining_candidates,
                new_remaining_block_sizes,
                current_block_number + 1,
                candidates_assigned_so_far + items_this_block,
            )

            assignments += (
                self.get_from_pmf_cache(max_possible_this_block, items_this_block)
                * suffixes
            )

        return assignments

    def _calculate_legal_assignments_aux(
        self,
        remaining_candidates,
        remaining_block_sizes,
        current_block_number,
        candidates_assigned_so_far,
    ):
        key = LegalAssignmentKey(
            remaining_candidates,
            remaining_block_sizes,
            current_block_number,
            candidates_assigned_so_far,
        )

        if not key.__hash__ in self.legal_assignment_cache:
            self.legal_assignment_cache[
                key.__hash__
            ] = self._find_legal_assignments_aux(
                remaining_candidates,
                remaining_block_sizes,
                current_block_number,
                candidates_assigned_so_far,
            )

        return self.legal_assignment_cache[key.__hash__]


class LegalAssignmentKey:
    """Utility class for the recursive fail prob"""

    def __init__(
        self,
        remaining_candidates,
        remaining_block_sizes,
        current_block_number,
        candidates_assigned_so_far,
    ):
        self.remaining_candidates = remaining_candidates
        self.remaining_block_sizes = remaining_block_sizes
        self.current_block_number = current_block_number
        self.candidates_assigned_so_far = candidates_assigned_so_far

    def __eq__(self, other):
        if self.remaining_candidates != other.remaining_candidates:
            return False
        if self.current_block_number != other.current_block_number:
            return False
        if self.candidates_assigned_so_far != other.candidates_assigned_so_far:
            return False
        if self.remaining_block_sizes != other.remaining_block_sizes:
            return False
        return True

    def __hash__(self):
        return (
            (self.remaining_candidates + len(self.remaining_block_sizes) << 16)
            + self.current_block_number
            + self.candidates_assigned_so_far
        )


class MTableFailProbPair:
    """
    Encapsulation of all parameters for the interim mtables
    """

    def __init__(self, k, p, alpha, fail_prob, mtable):
        self.k = k
        self.p = p
        self.alpha = alpha
        self.fail_prob = fail_prob
        self.mtable = mtable

    def mass_of_mtable(self):
        return self.mtable["m"].sum()

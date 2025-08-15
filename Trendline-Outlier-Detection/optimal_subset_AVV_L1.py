from sortedcontainers import SortedDict
import pandas as pd
from typing import List, Union, Type
from aggregations_mem import AggregationMem

class OptimalSubset_AVV_L1():
    def calc_new_col_violation(self, agg_value, group_values):
        """
        Calculate the new column violation for the given agg_value and group_values , under AVV and L1.
        :param agg_value: The aggregation value to check.
        :param group_values: List of tuples (agg_value, group_id) representing the groups.
        :return: The sum of violations (AVV) of new column.
        """
        if len(group_values) == 0:
            return 0
        sum_of_violations = 0
        for agg_value_group, _ in group_values:
            if agg_value_group > agg_value:
                sum_of_violations += (agg_value_group - agg_value)
        return sum_of_violations

    def update_H_with_pruning(self, F, H, group_id, tau=0, violation_vector='CVV'):
        """
        :param F: dictionary from agg value to count of remaining tuples (for group i)
        :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i-1, such that agg value of group i-1 = x)
        :param group_id: value of group by column that represents group i.
        :return:
        """
        H_original_copy = H.copy()  # make a copy of H to avoid modifying it while iterating
        agg_values = sorted(F.keys(), reverse=True)  # sort feasible aggregation values from large to small
        # for agg_value in range(len(F) - 1, 0, -1):
        for agg_value in agg_values:
            if F[agg_value] > 0:
                index = H.bisect_right(agg_value)
                largest_below_agg_value = H.keys()[index-1]
                candidate_repair_size = H[largest_below_agg_value][0] + F[agg_value]
                if (agg_value not in H) or (candidate_repair_size > H[agg_value][0]):
                    new_list = H[largest_below_agg_value][1].copy()
                    new_list.insert(0, (agg_value, group_id))
                    H[agg_value] = (candidate_repair_size, new_list)
        return H


    def update_H_no_pruning(self, F, H, group_id , tau=0 , violation_vector='AVV'):
        """
        :param F: dictionary from agg value to count of remaining tuples (for group i)
        :param H: sorted dict of "max agg value of all cols in {1..i-1}" (x) to: (count, [(agg_value, group_id)]) for groups 1..i-1, such that agg value of group i-1 = x)
        :param group_id: value of group by column that represents group i.
        :return:
        """
        H_original_copy = H.copy()  # make a copy of H to avoid modifying it while iterating
        agg_values = sorted(F.keys(), reverse=False)  # sort feasible aggregation values from large to small
        # add to agg_values the agg_values that come from H , only the [0] part of the tuple
        agg_values.extend([key[0] for key in H_original_copy.keys()])
        agg_values = sorted(set(agg_values), reverse=False)  # remove duplicates and sort again
        
        H_keys = sorted(H_original_copy.keys(), reverse=False) # keep frozen for the iteration
        F_keys = sorted(F.keys(), reverse=False) # keep frozen for the iteration

        for agg_value in agg_values:
            for budget in range(0, tau+1):
                previous_max_repair_size = 0
                previous_max_repair_size_2 = 0
                previous_max_repair_group_values = []
                previous_max_repair_group_values_2 = []
                candidate_repair_size = 0
                candidate_repair_size2 = 0
                # first case
                if agg_value in F:
                    H_keys_filtered = [k for k in H_keys if k[0] <= agg_value and k[1] == budget]
                    for key in H_keys_filtered:
                        repair_size = H_original_copy[key][0]
                        if repair_size > previous_max_repair_size:
                            previous_max_repair_size = repair_size
                            previous_max_repair_group_values = H_original_copy[key][1].copy()
                    candidate_repair_size = previous_max_repair_size + F[agg_value]

                # second case : iterate over F this time until agg_value
                for budget_tag in range(0, budget+1):
                    if (agg_value, budget_tag) not in H_original_copy: # if there is no such key in H, we cannot use it
                        continue
                    F_keys_filtered = [k for k in F_keys if (agg_value - (budget-budget_tag)) <= k < agg_value]
                    # while F_index < len(F_keys) and F_keys[F_index] < agg_value and F_keys[F_index] <= agg_value-(budget-budget_tag): # find the largest agg value in H that is smaller than agg_value
                    for key in F_keys_filtered:
                        if self.calc_new_col_violation(key , H_original_copy[(agg_value,budget_tag)][1])+ budget_tag != budget:
                            continue
                        repair_size2 = F[key] + H_original_copy[(agg_value,budget_tag)][0]
                        if repair_size2 > previous_max_repair_size_2:
                            previous_max_repair_size_2 = repair_size2
                            previous_max_repair_group_values_2 = H_original_copy[(agg_value,budget_tag)][1].copy()
                            previous_max_repair_group_values_2.insert(0,(key, group_id))
                candidate_repair_size2 = previous_max_repair_size_2
                    
                # Update according to the best of case1 , case2                
                if candidate_repair_size2 > candidate_repair_size:
                    if ((agg_value,budget) not in H) or (candidate_repair_size2 > H[agg_value,budget][0]):
                        new_list = previous_max_repair_group_values_2.copy()
                        H[agg_value,budget] = (candidate_repair_size2, new_list)
                if candidate_repair_size > candidate_repair_size2:
                    if ((agg_value,budget) not in H) or (candidate_repair_size > H[agg_value,budget][0]):
                        new_list = previous_max_repair_group_values.copy()
                        new_list.insert(0, (agg_value, group_id))
                        H[agg_value,budget] = (candidate_repair_size, new_list)
                if candidate_repair_size == candidate_repair_size2:
                    if ((agg_value,budget) not in H) or (candidate_repair_size > H[agg_value,budget][0]):
                        new_list = previous_max_repair_group_values.copy()
                        new_list.insert(0, (agg_value, group_id))
                        new_list.extend(previous_max_repair_group_values_2)
                        H[agg_value,budget] = (candidate_repair_size, new_list)
                if candidate_repair_size == 0 and candidate_repair_size2 == 0:
                    if (agg_value, budget) not in H:
                        H[agg_value, budget] = (0, [])    
        return H


    def prune_H(self, H, max_removed=None, sum_of_group_sizes=None):
        """
        If x1<=x2 and H(x1)>=H(x2), keep only x1, and prune x2.
        :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i, such that agg value of group i = x)
        """
        max_count = -1
        newH = {}
        for option in H.keys():
            if option > 0 and max_removed is not None:
                # compute removal from groups 1,.., i. If it's too large, no need to remember this option.
                if (sum_of_group_sizes - H[option][0]) > max_removed:
                    continue
            if H[option][0] > max_count:
                newH[option] = H[option]
                max_count = H[option][0]
        return SortedDict(newH)


    def prune_H_by_max_removed(self, H, max_removed, sum_of_group_sizes=None):
        """
        If x1<=x2 and H(x1)>=H(x2), keep only x1, and prune x2.
        :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i, such that agg value of group i = x)
        """
        newH = {}
        for option in H.keys():
            # compute removal from groups 1,.., i-1. If it's too large, no need to remember this option.
            if (sum_of_group_sizes - H[option][0]) > max_removed:
                continue
            newH[option] = H[option]
        return SortedDict(newH)


    def get_optimal_subset_F_first(self,
            df: pd.DataFrame,
            group_cols: Union[str, List[str]],
            agg_col: str,
            #agg_func_str: str,
            Agg: Type[AggregationMem],
            max_removed: int = None,
            prune_dp_by_max_removed: int = None,
            prune_h: bool = False,
            time_cutoff_seconds: int = None
    ) -> (pd.DataFrame, pd.DataFrame):
        print("AVV and Linf case")
        print(len(df))
        print("mem opt, F first")
        if max_removed is not None:
            print(f"prune agg pack: {max_removed}")
        if prune_dp_by_max_removed is not None:
            print(f"prune dp: {prune_dp_by_max_removed}")
        print(f"prune h: {prune_h}")
        df = df.loc[df[group_cols].notnull().all(axis=1)].reset_index(drop=True)
        print("agg result before repair:")
        print(df.groupby(group_cols)[agg_col].agg(['sum', 'count', 'mean', 'median', 'max']))

        output = {}

        H = SortedDict()
        # set all H[(0,x)] = (0, []) for all x in range(0, tau+1)
        # if tau > 0:
        #     for x in range(0, tau + 1):
        #         H[(0, x)] = (0, [])
        group_keys = []

        aggs = {}
        group_sizes = {}
        # First compute F (realizable aggregations and max subset size) for each group.
        for group_key, group_df in df.groupby(group_cols):  # groupby keys are sorted by default
            print(f"working on group: {group_key}")
            agg = Agg()
            output[group_key] = agg.compute_max_subset_sizes(group_df, agg_col, max_removed, time_cutoff_seconds)
            aggs[group_key] = agg
            group_keys.append(group_key)
            group_sizes[group_key] = len(group_df)
            
        # Next, compute the solution (main DP).
        sum_of_group_sizes = 0
        for group_key in group_keys:
            print(f"merging + pruning group: {group_key}")
            sum_of_group_sizes += group_sizes[group_key]
            if prune_h:
                H = self.update_H_with_pruning(output[group_key], H, group_key)
                H = self.prune_H(H, prune_dp_by_max_removed, sum_of_group_sizes) #TODO: there is a bug here when prune_dp_by_max_removed isn't None!
            else:
                H = self.update_H_no_pruning(output[group_key], H, group_key)
                if prune_dp_by_max_removed is not None:
                    H = self.prune_H_by_max_removed(H, prune_dp_by_max_removed, sum_of_group_sizes)

        ids_to_keep = []
        if prune_h:
            # We don't need to search for the best solution in H because of the pruning.
            # The solution with the largest x value will be the largest repair.
            largest_x = H.keys()[-1]
            print(f"largest_x: {largest_x}, H[largest_x]={H[largest_x]}")
            agg_values_and_group_keys = H[largest_x][1]
        else:
            # No pruning - search for the best solution in H
            best_repair_size = 0
            agg_values_and_group_keys = None
            for x in H:
                repair_size, group_values = H[x]
                if repair_size > best_repair_size:
                    best_repair_size = repair_size
                    agg_values_and_group_keys = group_values

        for agg_value, group_key in agg_values_and_group_keys:
            print(f"find subset for group {group_key} with agg value {agg_value}")
            ids_to_keep.extend(aggs[group_key].get_subset_for_value(agg_value))

        subset_df = df.iloc[ids_to_keep]
        removed_df = df.loc[~df.index.isin(ids_to_keep)]
        print("agg result after repair:")
        print(subset_df.groupby(group_cols)[agg_col].agg(['sum', 'count', 'mean', 'median', 'max']))
        #print(f"num_removed: {len(removed_df)}")

        return subset_df, removed_df

        # for agg_value, key in H[H.keys()[-1]][1]:
        #     items_in_key = list(get_subset_with_sum(vals[key], data[key], agg_value))
        #     for item in items_in_key:
        #         print((key[0], item[0]), item[1])
        #         needed_items[(key[0], item[0])] = item[1]
        #
        # indices_to_remove = []
        # for idx, row in df.iterrows():
        #     if ((row[group_cols[0]], row[agg_col]) in needed_items) and needed_items[
        #         (row[group_cols[0]], row[agg_col])] > 0:
        #         needed_items[(row[group_cols[0]], row[agg_col])] = needed_items[(row[group_cols[0]], row[agg_col])] - 1
        #     else:
        #         indices_to_remove.append(idx)
        # print(indices_to_remove)
        # df2 = df.drop(indices_to_remove).reset_index(drop=True)
        # df2.to_csv('df2_output.csv', index=False)
        # print(H[H.keys()[-1]])

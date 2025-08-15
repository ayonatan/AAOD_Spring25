from sortedcontainers import SortedDict
import pandas as pd
from typing import List, Union, Type
from aggregations_mem import AggregationMem

class OptimalSubset_AVV_Linf():

    def update_H_with_pruning(self, F, H, group_id, tau=0, violation_vector='CVV'):
        """
        :param F: dictionary from agg value to count of remaining tuples (for group i)
        :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i-1, such that agg value of group i-1 = x)
        :param group_id: value of group by column that represents group i.
        :return:
        """
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

    def build_entry_from_range(self, agg_df, start_age, end_age):
        # חיתוך טווח גילאים
        subset = agg_df.loc[start_age:end_age]

        # group_id מתחיל מ-1 לגיל ההתחלתי
        result_list = []
        group_id = 25
        for age in subset.index:
            agg_value = subset.loc[age, 'max']
            result_list.append((agg_value, (group_id,)))
            group_id += 1
        #reverse the list 
        result_list.reverse()
        # סכום ה-count בטווח
        total_count = subset['count'].sum()

        # המבנה הסופי
        return (total_count, result_list)


    def update_H_no_pruning(self, agg_df, F, H, group_id , tau=0 , violation_vector='AVV'):
        """
        :param F: dictionary from agg value to count of remaining tuples (for group i)
        :param H: sorted dict of "max agg value of all cols in {1..i-1}" (x) to: (count, [(agg_value, group_id)]) for groups 1..i-1, such that agg value of group i-1 = x)
        :param group_id: value of group by column that represents group i.
        :return:
        """
        H_original_copy= H.copy()  # make a copy of H to avoid modifying it while iterating
        agg_values = sorted(F.keys(), reverse=False)  # sort feasible aggregation values from large to small
        # add to agg_values the agg_values that come from H
        agg_values.extend(H_original_copy.keys())
        agg_values = sorted(set(agg_values), reverse=False)  # remove duplicates and sort again
        
        H_keys = sorted(H_original_copy.keys(), reverse=False) # keep frozen for the iteration
        F_keys = sorted(F.keys(), reverse=False) # keep frozen for the iteration

        print("agg_values = ", agg_values)
        # previous_max_repair_size = 0
        # previous_max_repair_size_2 = 0
        # previous_max_repair_group_values = []
        # previous_max_repair_group_values_2 = []

        for agg_value in agg_values:
            # print("H before update_H_no_pruning on group_id", group_id, "with agg_value", agg_value, "is:")
            # for key, value in H.items():
            #     print(f"  {key}: {value}")
            H_index = 0
            previous_max_repair_size = 0
            previous_max_repair_size_2 = 0
            previous_max_repair_group_values = []
            previous_max_repair_group_values_2 = []
            candidate_repair_size = 0
            candidate_repair_size2 = 0
            # first case
            if agg_value in F:
                while H_index < len(H_keys) and H_keys[H_index] <= agg_value: # find the largest agg value in H that is smaller than or equal to agg_value            
                    repair_size = H_original_copy[H_keys[H_index]][0]
                    if repair_size > previous_max_repair_size:
                        previous_max_repair_size = repair_size
                        previous_max_repair_group_values = H_original_copy[H_keys[H_index]][1].copy()
                    H_index += 1
                candidate_repair_size = previous_max_repair_size + F[agg_value]
                # previous_max_repair_group_values.insert(0, (agg_value, group_id))
            # if agg_value==18:
            #     print("H_keys = ", H_keys)
            #     print("F_keys = ", F_keys)
            #     print("agg_value = ", agg_value)
            #     print("previous_max_repair_size = ", previous_max_repair_size)
            #     print("previous_max_repair_group_values = ", previous_max_repair_group_values)
            #     print("candidate_repair_size = ", candidate_repair_size)
            #     print("--------------------")
            # second case : iterate over F this time until agg_value
            temp = []
            # print("F.keys for group_id", group_id, "is:", F.keys())
            F_index = 0
            if agg_value in H_original_copy:
                F_keys_filtered = [k for k in F_keys if (agg_value - tau) <= k < agg_value]
                for key in F_keys_filtered: # find the largest agg value in H that is smaller than agg_value
                    repair_size2 = F[key]
                    if repair_size2 > previous_max_repair_size_2:
                        previous_max_repair_size_2 = repair_size2
                        temp = [(key, group_id)].copy()
                # print("H[agg_value][0] = ", H[agg_value][0],"agg_value = ", agg_value, "previous_max_repair_size_2 = ", previous_max_repair_size_2)
                candidate_repair_size2 = H_original_copy[agg_value][0] + previous_max_repair_size_2
                # previous_max_repair_group_values_2 is H[agg_value][1].copy() + [(F_keys[F_index], group_id)]
                previous_max_repair_group_values_2 = H_original_copy[agg_value][1].copy()
                # if temp:
                #     previous_max_repair_group_values_2.insert(0, temp[0])

            # Update according to the best of case1 , case2  
            # print("agg_value = ", agg_value, "candidate_repair_size = ", candidate_repair_size, "candidate_repair_size2 = ", candidate_repair_size2)              
            if candidate_repair_size2 > candidate_repair_size:
                if (agg_value not in H) or (candidate_repair_size2 > H[agg_value][0]):
                    if temp:
                        previous_max_repair_group_values_2.insert(0, temp[0])
                    new_list = previous_max_repair_group_values_2.copy()
                    H[agg_value] = (candidate_repair_size2, new_list)
            if candidate_repair_size > candidate_repair_size2:
                if ((agg_value not in H) or (candidate_repair_size > H[agg_value][0])):
                    previous_max_repair_group_values.insert(0, (agg_value, group_id))
                    new_list = previous_max_repair_group_values.copy()
                    H[agg_value] = (candidate_repair_size, new_list)
            if candidate_repair_size2 == candidate_repair_size and candidate_repair_size > 0:
                if (agg_value not in H) or (candidate_repair_size2 > H[agg_value][0]):
                    new_list = previous_max_repair_group_values_2.copy()
                    H[agg_value] = (candidate_repair_size2, new_list)
                
            if candidate_repair_size2 == candidate_repair_size and candidate_repair_size == 0:
                if agg_value not in H:
                    H[agg_value] = (0, [])
        # print("H after update_H_no_pruning on group_id", group_id, "is:")
        # for key, value in H.items():
        #     print(f"  {key}: {value}")
        
        # H must have the original agg cols from group_id = 1 to group_id = i
        # loop on each group key that was added
        # the original agg value of group_key
        # print("group_id = ", group_id)
        # # ages is dict from age to group_id like {group_id: age}
        # ages = agg_df.index.to_series().to_dict()
        # entry = build_entry_from_range(agg_df, 25, ages[group_id[0]])
        # print("H after update_H_no_pruning on group_id", group_id, "is:")
        # for key, value in H.items():
        #     print(f"  {key}: {value}")

        # if entry not in H.values():
        #     # print(f"Error: entry {entry} from group {group_id} is not in H after update_H_no_pruning")
        #     # print("H keys: ", H.keys())
        #     # print("H values: ", H.values())
        #     # print("agg_df: ", agg_df)
        #     raise ValueError(f"entry {entry} from group {group_id} is not in H after update_H_no_pruning")
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
        agg_df = df.groupby(group_cols)[agg_col].agg(['sum', 'count', 'mean', 'median', 'max'])
        print(agg_df)

        output = {}

        H = SortedDict()
        H[0] = (0, [])  # first element is the amount of items, the second is the aggregation value in each key group
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
                H = self.update_H_no_pruning(agg_df, output[group_key], H, group_key)
                # check if in any of H elements there is a key that returns more than once
                # for elem in H:
                #     # H[elem][1] is a list of tuples (agg_value, group_id)
                #     # check if there are duplicates in the list of group_ids
                #     group_ids = [x[1] for x in H[elem][1]]
                #     if len(group_ids) != len(set(group_ids)):
                #         print(f"Error: there are duplicates in group_ids for elem {elem} in H: {H[elem][1]}")
                #         raise ValueError(f"Error: there are duplicates in group_ids for elem {elem} in H: {H[elem][1]}")
                    
                #sanity_check : after update_H_no_pruning, we should have the original columns in H
                # loop on each group key thas was added
                # the original agg value of group_key
                # for agg_value in output[group_key].keys():
                #     if agg_value not in H:
                #         print(f"Error: agg_value {agg_value} from group {group_key} is not in H after update_H_no_pruning")
                #         print("H keys: ", H.keys())
                #         print("output[group_key]: ", output[group_key])
                #         raise ValueError(f"agg_value {agg_value} from group {group_key} is not in H after update_H_no_pruning")
                # find the agg value of group_key in x
                
                
                # x = df.groupby(group_cols)[agg_col].agg(sum_agg='sum', count_agg='count', mean_agg='mean', median_agg='median', max_agg='max')
                # agg_value = x.loc[group_key, 'max_agg']  # or whatever metric you actually need
                # print(f"Checking agg_value {agg_value} for group {group_key}")
                # # print the solution in H with agg_value
                # if agg_value in H:
                #     print(f"Found solution in H for agg_value {agg_value}: {H[agg_value]}")
                # else:
                #     print(f"No solution found in H for agg_value {agg_value}")

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
            # print(f"find subset for group {group_key} with agg value {agg_value}")
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

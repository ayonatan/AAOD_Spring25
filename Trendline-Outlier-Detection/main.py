from matplotlib import pyplot as plt
from input_parser import parse_input
from optimal_subset_with_constraint import get_optimal_subset, get_optimal_subset_mem_opt, get_optimal_subset_pruning_mem_opt
from optimal_subset_with_constraints_pruning import get_optimal_subset_pruning
from registry import ALGO_MAP
import os,time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input_data = parse_input()
    # make a copy of the input data for debugging purposes with name "input_data_copy"
    print("The parsed data is: \n", input_data.df)

    AlgoClass = ALGO_MAP[(input_data.violation_vector, input_data.norm)]
    algo = AlgoClass()
    s = time.time()

    subset_df, removed_df = algo.get_optimal_subset_F_first(
        df=input_data.df,
        group_cols=input_data.group_cols,
        agg_col=input_data.agg_col,
        Agg=input_data.aggregation,
        max_removed=input_data.prune_aggpack_by_greedy,
        prune_dp_by_max_removed=input_data.prune_dp_by_greedy,
        prune_h=input_data.prune_h,
        time_cutoff_seconds=input_data.time_cutoff_seconds)

    print(f"Num removed tuples: {len(removed_df)}/{len(input_data.df)}")
    print(f"time: {time.time() - s}")
    
    # print("Optimal solution is: \n", subset_df) # print the optimal subset in the console
    # print("The removed tuples are: \n", removed_df) # print the removed tuples in the console

    #subset_df.to_csv(os.path.join(args.output_folder, f"dp_result-{input_data.orig_fname}.csv"), index=True)
    removed_df.to_csv(os.path.join(input_data.output_folder, f"dp_removed-{input_data.orig_fname}.csv"), index=True)    

    # PLOT ONLY THE AFTER REMOVAL AGGREGATION
    after_agg = subset_df.groupby(input_data.group_cols)[input_data.agg_col].agg(['sum', 'count', 'mean', 'median', 'max'])
    groups = after_agg.index.astype(str)  # גילאים כקטגוריות למספרים על הציר
    x = np.arange(len(groups)) * 1.2  # רווח בין העמודות

    width = 0.5

    plt.figure(figsize=(12, 6))
    plt.bar(x, after_agg['max'], width, color='orange', label='After Removal')

    plt.xticks(x, groups, rotation=45, ha='right')
    plt.xlabel('Age')
    plt.ylabel('Max Transaction Price')
    plt.title('H&M Dataset - Max Aggregation of Transaction Price grouped by Age (After Removal)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    #PLOT ONLY THE BEFORE AGGREGATION
    # before_agg = input_data.df.groupby(input_data.group_cols)[input_data.agg_col].agg(['sum', 'count', 'mean', 'median', 'max'])

    # groups = before_agg.index.astype(str)

    # # קישור שמות אגרגציה לפרמטר (לדוגמה 'MAX' -> 'max')
    # agg_name_to_metric = {
    #     'MAX': 'max',
    #     'MIN': 'min',
    #     'SUM': 'sum',
    #     'COUNT': 'count',
    #     'MEAN': 'mean',
    #     'MEDIAN': 'median',
    # }

    # agg_name = input_data.agg_name.upper()
    # metric = agg_name_to_metric.get(agg_name, 'sum')

    # import numpy as np
    # import matplotlib.pyplot as plt

    # x = np.arange(len(groups))  # מיקומי העמודות
    # width = 0.6

    # plt.figure(figsize=(12,6))
    # plt.bar(x, before_agg[metric], width, label='Before Repair', color='blue', alpha=0.7)
    # plt.xticks(x, groups, rotation=45, ha='right')
    # plt.xlabel(f'Group: {input_data.group_cols[0]}')  # ציר האיקס - קבוצת הגיל
    # plt.ylabel(f'{metric.capitalize()} Transaction Price')  # ציר הוואי - מקסימום מחיר טרנזקציה
    # plt.title(f'H&M Dataset - Aggregation: {metric.capitalize()} Before Repair by Group')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

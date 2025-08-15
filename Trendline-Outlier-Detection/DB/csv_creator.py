import csv

filename = "motivation_example_yonatan.csv"

rows = []

transaction_id = 2000000

# make values_arr as tuples array with age
values_arr = [(10.9 , 20), (13.0, 30), (15.4, 35), (16.5, 45), (18.9, 50), (19.6, 55), (21.7, 60)]
# multople each value in the array (the first element of the tuple) by 1000
values_arr = [(round(value * 100, 2), age) for value, age in values_arr]
for value, age in values_arr:
    for _ in range(2000):
        transaction_id += 1
        rows.append((transaction_id, age, value))
    
values_arr_2 = [(13.1,25),(11.5,25),(16.7,40),(16,40)]  
values_arr_2 = [(round(value * 100, 2), age) for value, age in values_arr_2] 
for value, age in values_arr_2:
    for _ in range(1000):
        transaction_id += 1
        rows.append((transaction_id, age, value))
    
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['TransactionID', 'age', 'price_int'])
    writer.writerows(rows)

print(f"{len(rows)} rows written to {filename}")

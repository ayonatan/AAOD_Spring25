import pandas as pd

# קלט ופלט
# may_june_july
input_file = r"C:\Users\yonat\OneDrive\שולחן העבודה\semester 8\Project Brit\Trendline-Outlier-Detection\DB\may_june_july_transactions_processed_lean.csv"
output_file = r"C:\Users\yonat\OneDrive\שולחן העבודה\semester 8\Project Brit\Trendline-Outlier-Detection\DB\may_june_july_transactions_processed_lean_sample.csv"

# קריאת הקובץ
df = pd.read_csv(input_file)

# בחירה אקראית של חצי מהשורות (ללא החזרה)
df_half = df.sample(frac=0.0001, random_state=42)  # ניתן לשנות את seed אם רוצים שונות

# כתיבה לקובץ חדש
df_half.to_csv(output_file, index=False)

print(f"Saved {len(df_half)} rows to '{output_file}'")

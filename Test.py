import pandas as pd
import gc
df = pd.read_csv("https://raw.githubusercontent.com/CoolSmit007/Product-Recommender/main/ProcessedRatings_110.csv", error_bad_lines=False)
df.to_csv("ProcessedRatings_110.csv",index=False)
del(df)
gc.collect()
print(f"😀 SQL vs Pandas 비교하기 👍")

import pandas as pd
import numpy as np

url = "https://raw.github.com/pandas-dev/pandas/master/pandas/tests/io/data/csv/tips.csv"

tips = pd.read_csv(url)

print(f"✅ 불러온 데이터 상위 5개 데이터 보기")
print(tips.head(5))

print(f"✅ SELECT")
print(tips[["total_bill", "tip", "smoker", "time"]])
# SELECT total_bill, tip, smoker, time
# FROM tips;

print(f"✅ LIMIT")
print(tips.head(10))
# SELECT *
# FROM tips
# LIMIT 10;

print(f"✅ LIMIT OFFSET")
print(tips.iloc[3:8, :])
# SELECT *
# FROM tips
# LIMIT 5 OFFSET 3;

print(f"✅ UNIQUE")
print(tips.day.unique())
# SELECT DISTINCT day
# FROM tips;

print(tips.day.nunique())
# SELECT COUNT(DISTINCT day)
# FROM tips;

print(f"✅ WHERE")
print(tips[tips.sex ==  "Female"][["tip", "smoker"]][:5])
# SELECT tip, smoker
# FROM tips
# WHERE sex = 'Female'
# LIMIT 5; 

print(tips[(tips["time"] == "Diner") & (tips["tip"] > 5.00)])
# SELECT *
# FROM tips
# WHERE time = 'Dinner' AND tip > 5.00;

print(tips[(tips["size"] >= 5) | (tips["total_bill"] > 45)])
# SELECT *
# FROM tips
# WHERE size >= 5 OR total_bill > 45;

print(f"✅ DIVISION")
tips["tip_rate"] = tips["tip"] / tips["total_bill"]
print(tips["tip_rate"])
# SELECT tip/total_bill AS tip_rate
# FROM tips;

print(f"✅ VALUE COUNTS")
print(tips.day.value_counts())
# SELECT day, COUNT(day) AS 'count'
# FROM tips
# GROUP BY day;

print(f"✅ GROUP BY")
print(tips.groupby('sex').count())
# SELECT sex, COUNT(*)
# FROM tips
# GROUP BY sex;

print(tips.groupby("day").agg({"tip": np.mean, "day": np.size}))
# SELECT day, AVG(tip), COUNT(*)
# FROM tips
# GROUP BY day;

print(tips.groupby(["smoker", "day"].agg({"tips" : [np.size, np.mean]})))
# SELECT smoker, day, COUNT(*), AVG(tip)
# FROM tips
# GROUP BY smoker, day;

print(f"✅ Null Value")
frame = pd.DataFrame({"col1": ["A", "B", np.NaN, "C", "D"], "col2": ["F", np.NaN, "G", "H", "I"]})
print(frame)
print(frame[frame["col2"].isna()])
# SELECT *
# FROM frame
# WHERE col2 IS NULL;

print(frame[frame["col1"].notna()])
# SELECT *
# FROM frame
# WHERE col1 IS NOT NULL;

print(f"✅ JOIN")
df1 = pd.DataFrame({"key": ["A", "B", "C", "D"], "value": np.random.randn(4)})
print(df1)
df2 = pd.DataFrame({"key": ["B", "D", "D", "E"], "value": np.random.randn(4)})
print(df2)

print(f"✅ INNER JOIN")
print(pd.merge(df1, df2, on="key"))
# SELECT * 
# FROM df1
# INNER JOIN df2 
# ON df1.key = df2.key;

print(f"✅ LEFT JOIN")
print(pd.merge(df1, df2, on="key", how="left"))
# SELECT * 
# FROM df1
# LEFT OUTER JOIN df2
# ON df1.key= df2.key;

print(f"✅ OUTER JOIN")
print(pd.merge(df1, df2, on="key", how="outer"))
# SELECT * 
# FROM df1
# FULL OUTER JOIN df2
# ON df1.key = df2.key;

print(f"✅ CONCAT")
df1 = pd.DataFrame({"city": ["Chicago", "San Francisco", "New York City"], "rank": range(1, 4)})
print(df1)
df2 = pd.DataFrame({"city": ["Chicago", "Boston", "Los Angeles"], "rank": [1, 4, 5]})
print(df2)
print(pd.concat([df1, df2]))
# SELECT city, rank
# FROM df1
# UNION ALL
# SELECT city, rank
# FROM df2

print(pd.concat([df1, df2]).drop_duplicates())
# SELECT city, rank
# FROM df1
# UNION
# SELECT city, rank
# FROM df2;
print()

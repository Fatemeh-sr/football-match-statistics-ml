import pandas as pd

df = pd.read_csv("epl_final.csv")

# print(df.info())

features = df[
    [
        "FullTimeHomeGoals",
        "FullTimeAwayGoals",
        "HomeShots",
        "AwayShots",
        "HomeShotsOnTarget",
        "AwayShotsOnTarget",
    ]
]

"""
'H'	برد میزبان (Home Win)
'D'	مساوی (Draw)
'A'	برد مهمان (Away Win)
"""
# FullTimeResult   --> Target    .. H = 3 , D = 1 , A = 0

result_mapping = {"H": 3, "D": 1, "A": 0}
target = df["FullTimeResult"].map(result_mapping)

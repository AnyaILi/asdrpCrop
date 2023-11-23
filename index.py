import pandas as pd
df = pd.DataFrame({'Number': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}, 
                   'Power 2': {0: 1, 1: 4, 2: 9, 3: 16, 4: 25}, 
                   'Power 3': {0: 1, 1: 8, 2: 27, 3: 64, 4: 125}}) 
  
# view dataframe 
print("Initial dataframe")
print(df)

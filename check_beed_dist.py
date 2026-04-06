import pandas as pd
df = pd.read_csv('/Users/abhinavbhargava/Downloads/Capstone/BEED_Data.csv')
print("Total Class distribution:", df['y'].value_counts().to_dict())
chunk_size = len(df) // 4
print("Chunk 1:", df['y'].iloc[:chunk_size].value_counts().to_dict())
print("Chunk 2:", df['y'].iloc[chunk_size:chunk_size*2].value_counts().to_dict())
print("Chunk 3:", df['y'].iloc[chunk_size*2:chunk_size*3].value_counts().to_dict())
print("Chunk 4:", df['y'].iloc[chunk_size*3:].value_counts().to_dict())

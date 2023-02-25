from nhanes.load import load_NHANES_data

data_df = load_NHANES_data()

print(data_df.columns.values)

#data = data_df[['StandingHeightCm','WeightKg','Gender']]
data = data_df[['StandingHeightCm','WeightKg']]

data.reset_index(drop=True,inplace=True)
data.index.name="ID"
data=data.dropna(how='any')
#print(data[data['WeightKg'] == ''])
print(data[70:80])

data.to_csv("./data.csv",header=False,index=False)

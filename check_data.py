import pandas as pd

df = pd.read_csv('exported_data_all/formatted_data_all.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date'])

print('Checking factor completeness by date...')
for d in sorted(df['date'].unique(), reverse=True)[:10]:
    df_d = df[df['date'] == d]
    valid = df_d['sales_to_price_ratio'].notna().sum()
    print(f'{d.date()}: {valid}/{len(df_d)} stocks with sales_to_price_ratio')

import akshare as ak

print("Searching for Shenwan (sw) industry/index APIs...")
for attr in dir(ak):
    if 'sw' in attr.lower() and ('index' in attr.lower() or 'industry' in attr.lower() or 'board' in attr.lower()):
        print(attr)

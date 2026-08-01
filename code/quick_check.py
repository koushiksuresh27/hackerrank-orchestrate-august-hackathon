import pandas as pd

df = pd.read_csv("../dataset/output.csv")
samples = pd.read_csv("../dataset/sample_messages.csv")
messages = pd.read_csv("../dataset/messages.csv")

merged_lookup = samples.merge(
    messages[['message_id', 'user_id', 'created_at']],
    on=['user_id', 'created_at'],
    how='left'
)

# Use message_id_y (the real msg_XXX id)
merged_lookup = merged_lookup.rename(columns={'message_id_y': 'real_message_id'})
print("Matched:", merged_lookup['real_message_id'].notna().sum(), "/ 30")

# Join with output predictions
matched = merged_lookup[merged_lookup['real_message_id'].notna()].copy()
merged = matched.merge(
    df,
    left_on='real_message_id',
    right_on='message_id',
    suffixes=('_expected', '_got')
)

print(f"\n=== ACCURACY vs {len(merged)} MATCHED SAMPLES ===")
action_match = (merged['action_expected'] == merged['action_got']).sum()
type_match = (merged['message_type_expected'] == merged['message_type_got']).sum()
print(f"Action accuracy: {action_match}/{len(merged)}")
print(f"Message type accuracy: {type_match}/{len(merged)}")

print("\n=== MISMATCHES ===")
mismatches = merged[merged['action_expected'] != merged['action_got']]
for _, row in mismatches.iterrows():
    print(f"[{row.real_message_id}] expected={row.action_expected} got={row.action_got} | type={row.message_type_expected} vs {row.message_type_got}")
    print(f"  reason: {row.reason_got}")
    print()

print("\n=== OUTPUT DISTRIBUTION ===")
print("Action:", df['action'].value_counts().to_dict())
print("Type:", df['message_type'].value_counts().to_dict())
print("Confidence:", df['confidence'].min(), df['confidence'].max(), round(df['confidence'].mean(),2))
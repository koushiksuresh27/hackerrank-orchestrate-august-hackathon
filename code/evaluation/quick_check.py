import pandas as pd

df = pd.read_csv("../dataset/output.csv")
samples = pd.read_csv("../dataset/sample_messages.csv")

print("=== YOUR OUTPUT DISTRIBUTION ===")
print("Action:", df['action'].value_counts().to_dict())
print("Type:", df['message_type'].value_counts().to_dict())
print("Confidence:", df['confidence'].min(), df['confidence'].max(), round(df['confidence'].mean(),2))
print("Empty reasons:", df['reason'].isna().sum())
print()

print("=== SAMPLE EXPECTED DISTRIBUTION ===")
print("Action:", samples['action'].value_counts().to_dict())
print("Type:", samples['message_type'].value_counts().to_dict())
print("Confidence:", samples['confidence'].min(), samples['confidence'].max(), round(samples['confidence'].mean(),2))
print()

print("=== 10 RANDOM OUTPUT ROWS ===")
for _, row in df.sample(10).iterrows():
    print(f"[{row.message_id}] {row.action} | {row.message_type} | conf={row.confidence}")
    print(f"  reason: {row.reason}")
    print(f"  evidence: {row.evidence_message_ids}")
    print()

print("=== SAMPLE REASON STYLE (for comparison) ===")
for _, row in samples.head(5).iterrows():
    print(f"[{row.message_id}] {row.action} | {row.message_type} | conf={row.confidence}")
    print(f"  reason: {row.reason}")
    print()
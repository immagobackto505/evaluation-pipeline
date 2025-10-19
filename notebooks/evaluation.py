import pandas as pd
import evaluate, sacrebleu
from tqdm import tqdm
import os
import glob
from google import genai

Instruction = pd.read_csv('Thai_Chinese_Dataset.csv')

# GET RESPONSES FROM GEMINI-2.5-FLASH

print("Generating responses using Gemini-2.5-flash...")
client = genai.Client(api_key='AIzaSyBVpv0baXVr9WBQ7scPkgpQx71MwKvFuxQ')

start_index = 0
batch_num = 1

for index, row in tqdm(Instruction.iterrows()):
    
    '''
    This loop generates responses for each instruction in the Instruction DataFrame using the Gemini-2.5-flash model.
    It saves the responses in batches of 10 to CSV files in the 'respond_batch' directory.

    '''

    # print(f"Processing row {index} \n")

    prompt = f"""You are an intelligent language model. 
    Follow the instruction carefully and respond concisely.

    Instruction: {row['instruction']}
    Input: "{row['input']}"
    Output:"""

    # print(prompt, '\n')

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    )
    
    Instruction.loc[index, 'respond'] = response.text

    if (index+1) % 10 == 0 or (index+1) == len(Instruction):
        print(f'BATCH {batch_num} SAVE : from {start_index} to {index}')
        batch_df = Instruction.iloc[start_index:index+1]
        os.makedirs("respond_batch", exist_ok=True)
        batch_df.to_csv(f"respond_batch/batch_{batch_num}.csv", index=True)
        batch_num += 1
        start_index = index+1

# combine the bacthes into one csv file
files = glob.glob("respond_batch/batch_*.csv")
respond_df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
respond_df.to_csv("outputs/responded_dataset.csv", index=False)
print("All batches combined and saved to 'outputs/responded_dataset.csv'.")

# EVALUATION

print("Evaluating responses...")
df = pd.read_csv('outputs/responded_dataset.csv')

# chrF++ per-sample (solid for ZH/TH)
chrf = evaluate.load("chrf")
df["chrf"] = [
    chrf.compute(predictions=[h], references=[[r]])["score"]
    for h, r in tqdm(list(zip(df["respond"], df["ref"])),
                     total=len(df), desc="chrF")
]

# sentence BLEU
df["bleu_sent"] = [
    sacrebleu.sentence_bleu(h, [r]).score
    for h, r in tqdm(list(zip(df["respond"], df["ref"])),
                     total=len(df), desc="BLEU (sent)")
]

# corpus BLEU
corpus_bleu = sacrebleu.corpus_bleu(df["respond"].tolist(), [df["ref"].tolist()]).score

# Optional semantic similarity
try:
    bs = evaluate.load("bertscore")
    df["bertscore_f1"] = bs.compute(
        predictions=df["respond"].tolist(),
        references=df["ref"].tolist(),
        lang="th"  # language of the hypothesis strings
    )["f1"]
except Exception as e:
    print("Skipping BERTScore (install torch + bert-score to enable). Reason:", e)

print("\n=== Qwen corpus summary ===")
print(f"chrF++ avg     : {df['chrf'].mean():.3f}")
print(f"BLEU (avg)     : {df['bleu_sent'].mean():.3f}")
print(f"BLEU (corpus)  : {corpus_bleu:.3f}")
if "bertscore_f1" in df:
    print(f"BERTScore F1   : {df['bertscore_f1'].mean():.4f}")

df.to_csv('outputs/evaluations.csv')
print("Evaluation results saved to 'outputs/evaluations.csv'.")

# SUMMARY CSV

print("Generating evaluation summary...")
df = pd.read_csv('outputs/evaluations.csv')
df = df.copy()[['type','chrf', 'bleu_sent', 'bertscore_f1']]

# add domain column
df['domain'] = df['type'].str.replace('.json', '').str.replace('-', '_').str.split('_').str[0]
df['domain'] = df['domain'].str.title().replace({
    'word': 'Word_Alignment',
    'Partial': 'Partial_Translation'
})

df.groupby('domain')[['chrf', 'bleu_sent', 'bertscore_f1']].agg(['mean', 'std'])
df.to_csv('outputs/evaluations_summary.csv')
print("Evaluation summary saved to 'outputs/evaluations_summary.csv'.")
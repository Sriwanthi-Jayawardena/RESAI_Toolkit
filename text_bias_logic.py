import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT for full pipeline
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

male_pronouns = ['he', 'him', 'his']
female_pronouns = ['she', 'her']

# ====== Full Pipeline ======
def get_word_embedding(word):
    tokens = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze(0).numpy()

def get_sentence_embedding(sentence):
    tokens = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze(0).numpy()

def run_text_bias_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    total_sentences = len(df)

    # Compute PMI
    P_pronoun = df['pronoun'].value_counts() / total_sentences
    P_occupation = df['occupation'].value_counts() / total_sentences
    joint = df.groupby(['occupation', 'pronoun']).size().reset_index(name='count')
    joint['P_joint'] = joint['count'] / total_sentences

    def compute_pmi(joint_val, occ, pro):
        p_occ = P_occupation.get(occ, 0)
        p_pro = P_pronoun.get(pro, 0)
        if joint_val > 0 and p_occ > 0 and p_pro > 0:
            return np.log2(joint_val / (p_occ * p_pro))
        return 0

    joint['PMI'] = joint.apply(lambda row: compute_pmi(row['P_joint'], row['occupation'], row['pronoun']), axis=1)

    # Aggregate PMI
    pmi_df = pd.DataFrame(df['occupation'].unique(), columns=['occupation'])
    pmi_df['PMI_male'] = pmi_df['occupation'].apply(
        lambda occ: joint[(joint['occupation'] == occ) & (joint['pronoun'].isin(male_pronouns))]['PMI'].mean()
    )
    pmi_df['PMI_female'] = pmi_df['occupation'].apply(
        lambda occ: joint[(joint['occupation'] == occ) & (joint['pronoun'].isin(female_pronouns))]['PMI'].mean()
    )
    pmi_df['PMI_Bias'] = pmi_df['PMI_male'] - pmi_df['PMI_female']

    # Cosine similarity
    male_emb = {w: get_word_embedding(w) for w in male_pronouns}
    female_emb = {w: get_word_embedding(w) for w in female_pronouns}

    cos_results = []
    for occ in pmi_df['occupation']:
        occ_emb = get_word_embedding(occ)
        cos_m = np.mean([cosine_similarity([occ_emb], [male_emb[w]])[0][0] for w in male_pronouns])
        cos_f = np.mean([cosine_similarity([occ_emb], [female_emb[w]])[0][0] for w in female_pronouns])
        cos_results.append({'occupation': occ, 'Cosine_Bias': cos_m - cos_f})
    cosine_df = pd.DataFrame(cos_results)

    # Contextual bias
    context_bias = []
    for occ in pmi_df['occupation']:
        male_sents = df[(df['occupation'] == occ) & (df['pronoun'].isin(male_pronouns))]['sentence'].tolist()
        female_sents = df[(df['occupation'] == occ) & (df['pronoun'].isin(female_pronouns))]['sentence'].tolist()

        if not male_sents or not female_sents:
            dist = np.nan
        else:
            m_vecs = np.array([get_sentence_embedding(s) for s in male_sents])
            f_vecs = np.array([get_sentence_embedding(s) for s in female_sents])
            dist = np.linalg.norm(m_vecs.mean(axis=0) - f_vecs.mean(axis=0))

        context_bias.append({'occupation': occ, 'Context_bias': dist})
    context_df = pd.DataFrame(context_bias)

    # Final CABM Score
    merged = pmi_df.merge(cosine_df, on='occupation').merge(context_df, on='occupation')
    merged['Gender_Bias_Score'] = (
        0.58328 * merged['PMI_Bias'] +
        0.15438 * merged['Cosine_Bias'] +
        0.26234 * merged['Context_bias']
    )

    # Assign Category
    merged['Bias_Category'] = merged['Gender_Bias_Score'].apply(label_bias)
    return merged[['occupation', 'Gender_Bias_Score', 'Bias_Category']]

# ====== Lightweight CABM from Features Only ======
def compute_score_from_features(csv_path):
    df = pd.read_csv(csv_path)

    if not {'occupation', 'PMI_Bias', 'Cosine_Bias', 'Context_bias'}.issubset(df.columns):
        raise ValueError("CSV must contain: occupation, PMI_Bias, Cosine_Bias, Context_bias")

    df['Gender_Bias_Score'] = (
        0.58328 * df['PMI_Bias'] +
        0.15438 * df['Cosine_Bias'] +
        0.26234 * df['Context_bias']
    )
    df['Bias_Category'] = df['Gender_Bias_Score'].apply(label_bias)
    return df[['occupation', 'Gender_Bias_Score', 'Bias_Category']]

# Shared categorization logic
def label_bias(score):
    if score < -0.1:
        return 'Female-Biased'
    elif score > 0.1:
        return 'Male-Biased'
    return 'Neutral'

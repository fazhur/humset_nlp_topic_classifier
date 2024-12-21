import os
import zipfile
import pandas as pd
import numpy as np
import torch

list_topics = ['At risk->Risk and vulnerabilities',
 'Capacities & response->International response',
 'Capacities & response->National response',
 'Capacities & response->People reached/response gaps',
 'Casualties->Dead',
 'Casualties->Injured',
 'Context->Demography',
 'Context->Economy',
 'Context->Environment',
 'Context->Legal & policy',
 'Context->Politics',
 'Context->Security & stability',
 'Context->Socio cultural',
 'Covid-19->Cases',
 'Covid-19->Deaths',
 'Covid-19->Prevention campaign',
 'Covid-19->Restriction measures',
 'Covid-19->Testing',
 'Covid-19->Vaccination',
 'Displacement->Push factors',
 'Displacement->Type/numbers/movements',
 'Humanitarian access->Physical constraints',
 'Humanitarian access->Relief to population',
 'Humanitarian conditions->Coping mechanisms',
 'Humanitarian conditions->Living standards',
 'Humanitarian conditions->Number of people in need',
 'Humanitarian conditions->Physical and mental well being',
 'Impact->Driver/aggravating factors',
 'Impact->Impact on people',
 'Impact->Impact on systems, services and networks',
 'Impact->Number of people affected',
 'Information and communication->Communication means and preferences',
 'Information and communication->Knowledge and info gaps (hum)',
 'Information and communication->Knowledge and info gaps (pop)',
 'Priority interventions->Expressed by humanitarian staff',
 'Priority needs->Expressed by humanitarian staff',
 'Priority needs->Expressed by population',
 'Shock/event->Hazard & threats',
 'Shock/event->Type and characteristics',
 'Shock/event->Underlying/aggravating factors'
]

def decompress_and_read_files(archive_path, output_dir="archive_dir"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(output_dir)

    files_data = []

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                files_data.append({"file_name": file, "text": text})

    df = pd.DataFrame(files_data)
    return df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def svm_based(data, clf, embedding_model):
    results = []

    topic_embeddings = {
        topic: embedding_model.encode(topic) for topic in list_topics
    }

    for _, row in data.iterrows():
        file_name = row["file_name"]
        full_text = row["text"]
        predicted_topics = []

        for text_segment in full_text.split('~~'):
            text_emb = embedding_model.encode(text_segment)

            for topic, topic_emb in topic_embeddings.items():
                concat_emb = np.concatenate([text_emb, topic_emb])
                logits = clf.decision_function([concat_emb])[0]
                probability = sigmoid(logits)

                if probability >= 0.75:
                    predicted_topics.append(topic)

        if not predicted_topics:
            results.append([file_name, full_text, "NA"])
        else:
            for topic in set(predicted_topics):
                results.append([file_name, full_text, topic])
    res_df = pd.DataFrame(results, columns=["file_name", "text", "topic"])
    res_df.drop_duplicates()
    return res_df


def get_relevant_labels(input, model, tokenizer, threshold=0.5):
    topic_mapping = {'Shock/event->Hazard & threats': 0,
    'At risk->Risk and vulnerabilities': 1,
    'Priority needs->Expressed by humanitarian staff': 2,
    'Humanitarian conditions->Number of people in need': 3,
    'Covid-19->Restriction measures': 4,
    'Shock/event->Underlying/aggravating factors': 5,
    'Displacement->Push factors': 6,
    'Humanitarian conditions->Coping mechanisms': 7,
    'Context->Security & stability': 8,
    'Context->Socio cultural': 9,
    'Casualties->Injured': 10,
    'Covid-19->Prevention campaign': 11,
    'Priority interventions->Expressed by humanitarian staff': 12,
    'Context->Economy': 13,
    'Displacement->Type/numbers/movements': 14,
    'Covid-19->Vaccination': 15,
    'Information and communication->Communication means and preferences': 16,
    'Context->Demography': 17,
    'Casualties->Dead': 18,
    'Capacities & response->People reached/response gaps': 19,
    'Capacities & response->National response': 20,
    'Humanitarian conditions->Living standards': 21,
    'Priority needs->Expressed by population': 22,
    'Impact->Impact on people': 23,
    'Impact->Number of people affected': 24,
    'Covid-19->Testing': 25,
    'Information and communication->Knowledge and info gaps (pop)': 26,
    'Context->Legal & policy': 27,
    'Context->Environment': 28,
    'Humanitarian conditions->Physical and mental well being': 29,
    'Capacities & response->International response': 30,
    'Context->Politics': 31,
    'Impact->Driver/aggravating factors': 32,
    'Information and communication->Knowledge and info gaps (hum)': 33,
    'Covid-19->Cases': 34,
    'Humanitarian access->Relief to population': 35,
    'Impact->Impact on systems, services and networks': 36,
    'Covid-19->Deaths': 37,
    'Humanitarian access->Physical constraints': 38,
    'Shock/event->Type and characteristics': 39}
    inversed_topic_mapping = {v: k for k, v in topic_mapping.items()}
    
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: inputs[key] for key in inputs}
    outputs = model(**inputs)
    logits = outputs.logits
    sigmoid_outputs = torch.sigmoid(logits)
    relevant_labels = [inversed_topic_mapping[i] for i, value in enumerate(sigmoid_outputs[0]) if value > threshold]
    return relevant_labels

    
def bert_based(data, bert_model, tokenizer):
    results = []
    for _, row in data.iterrows():
        file_name = row["file_name"]
        full_text = row["text"]
        predicted_topics = []

        for text_segment in full_text.split('~~'):
            predicted_topics.extend(get_relevant_labels(text_segment, bert_model, tokenizer, 0.5))
        
        if not predicted_topics:
            results.append([file_name, full_text, "NA"])
        else:
            for topic in set(predicted_topics):
                results.append([file_name, full_text, topic])

    res_df = pd.DataFrame(results, columns=["file_name", "text", "topic"])
    res_df.drop_duplicates()
    return res_df
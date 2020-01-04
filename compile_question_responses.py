import json 
from tqdm import tqdm 

MIN_NUM_WORDS = 5 

def check_question_response(response): 
    

    if '?' in response and len(response.split(' ')) > MIN_NUM_WORDS: 
        return True
    else:
        return False 

    return True 


with open('dailydialog_formatted.json', 'r') as f: 
    dd = json.load(f) 

with open("cornell_train_formatted.json", 'r') as f: 
    cornell = json.load(f) 

with open("persona_chat.json", 'r') as f: 
    persona = json.load(f)

# only dataset not in same format
with open('yesands_train_iter4.json', 'r') as f:
    yesands = json.load(f) 

reformatted_yesands = [] 
all_samples = [] 
for k, v in yesands['yesands'].items(): 
    all_samples += v

# formatting custom dataset in the same format as the ConvAI dataset used in original repo
for idx, yesand, in enumerate(all_samples): 
    instance = {"personality": "", "utterances": []}
    utterance = {"history": [yesand['p']],
                    "candidates": [all_samples[(idx+1)%len(all_samples)]['r'], yesand['r']]}
    instance["utterances"].append(utterance)
    reformatted_yesands.append(instance)


all_data = dd + cornell + persona + reformatted_yesands 

filtered_data = [] 
for instance in tqdm(all_data): 
    new_instance = {
        'personality': instance['personality'], 
        'utterances': [] 
    }
    for utt in instance['utterances']: 
        if check_question_response(utt['candidates'][-1]): 
            new_instance['utterances'].append(utt)
    
    if len(new_instance['utterances']) > 0: 
        filtered_data.append(new_instance)


print(filtered_data[0])
print(f"Number of training instances: {len(filtered_data)}")

with open(f"question_responses_longer_than{MIN_NUM_WORDS}.json", 'w') as f:
    json.dump(filtered_data, f, indent=4)




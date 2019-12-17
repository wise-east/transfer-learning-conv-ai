import json 
import pandas as pd 
from argparse import ArgumentParser
from pathlib import Path 

def get_df(json): 

    as_list = json['cornell'] + json['spont']
    queries = [item['p'] for item in as_list] 
    replies = [item['pred'] for item in as_list]

    df = pd.DataFrame({
        'id': list(range(len(queries))),
        'query': queries, 
        'reply': replies,
        'label': [-1]*len(replies)
    })

    return df

def main(): 

    parser = ArgumentParser() 
    parser.add_argument('--input', type=str, required=True, help="Input file to convert from json to csv")

    args = parser.parse_args() 

    with open(args.input, 'r') as f: 
        data = json.load(f) 

    df = get_df(data)

    save_path = (Path(args.input).absolute().parent / Path(args.input).stem).with_suffix('.csv')
    print(save_path)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
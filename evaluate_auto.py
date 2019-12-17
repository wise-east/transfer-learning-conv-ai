import nltk 
from argparse import ArgumentParser
import json

def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(nltk.ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)

def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

def main(): 

    parser = ArgumentParser() 
    parser.add_argument("--fp", type=str, help="identify the filepath that contains the predictions and gold output")

    args = parser.parse_args() 

    with open(args.fp, 'r') as f: 
        preds = json.load(f) 

    yesands = preds['spont'] + preds['cornell']
    refs = [[ya['r'].lower().split()] for ya in yesands]
    preds = [ya['pred'].lower().split() for ya in yesands]

    bleu2 = nltk.translate.bleu_score.corpus_bleu(refs, preds, [2])
    bleu4 = nltk.translate.bleu_score.corpus_bleu(refs, preds, [4])
    
    nist2 = nltk.translate.nist_score.corpus_nist(refs, preds, 2)
    nist4 = nltk.translate.nist_score.corpus_nist(refs, preds, 4)

    dist1 = distinct_n_corpus_level(preds, 1)
    dist2 = distinct_n_corpus_level(preds, 2)


    refs_sent = [ya['r'] for ya in yesands]
    preds_sent = [ya['pred'] for ya in yesands] 
    meteor_total = 0 
    for i in range(len(refs_sent)): 
        meteor_total += nltk.translate.meteor_score.meteor_score([refs_sent[i]], preds_sent[i])

    meteor_avg = meteor_total / len(refs_sent)

    avg_ref_len = sum([len(ref[0]) for ref in refs]) / len(refs)
    avg_preds_len = sum(len(pred) for pred in preds) / len(preds)

    print(f"\
            BLEU2: {bleu2}\t BLEU4: {bleu4}\t\n \
            NIST2: {nist2}\t NIST4: {nist4}\t\n \
            METEOR: {meteor_avg}\t\n \
            DIST1: {dist1}\t DIST2: {dist2}\t\n \
            AVG REF LEN: {avg_ref_len}\t AVG PRED LEN: {avg_preds_len}\t \
    ")
    # print(bleu2, bleu4)
    # print(nist2, nist4)
    # print(meteor_avg)
    # print(dist1, dist2)
    # print(avg_ref_len, avg_preds_len)



    return 


if __name__ == "__main__":
    main()
    # bleu2 = nltk.translate.bleu_score.sentence_bleu([sample['r'].lower().split()], out_text.lower().split(), [2])
    # bleu4 = nltk.translate.bleu_score.sentence_bleu([sample['r'].lower().split()], out_text.lower().split(), [4])

    # nist2 = nltk.translate.nist_score.sentence_nist([sample['r'].lower().split()], out_text.lower().split(), 2)
    # nist4 = nltk.translate.nist_score.sentence_nist([sample['r'].lower().split()], out_text.lower().split(), 4)

    # meteor = nltk.translate.meteor_score.meteor_score([sample['r']], out_text)

    # print(f"bleu2: {bleu2}, bleu4: {bleu4}, nist2: {nist2}, nist4: {nist4} meteor: {meteor}")
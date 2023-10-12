import codecs
from text_unidecode import unidecode
from typing import Tuple
import re
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
from typing import List
from collections import Counter
from nltk.corpus import stopwords

import spacy
from tqdm import tqdm
tqdm.pandas()


contractions = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

class Preprocessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.STOP_WORDS = set(stopwords.words('english'))
        self.spacy_ner_model = spacy.load('en_core_web_sm',)
        
    def count_text_length(self, df: pd.DataFrame, col:str) -> pd.Series:
        """ text length """
        tokenizer=self.tokenizer
        return df[col].progress_apply(lambda x: len(tokenizer.encode(x)))

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """        
        def check_is_stop_word(word):
            return word in self.STOP_WORDS
        
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))
            
    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int):
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        return len(common_ngrams)
    
    def ner_overlap_count(self, row, mode:str):
        model = self.spacy_ner_model
        def clean_ners(ner_list):
            return set([(ner[0].lower(), ner[1]) for ner in ner_list])
        prompt = model(row['prompt_text'])
        summary = model(row['text'])

        if "spacy" in str(model):
            prompt_ner = set([(token.text, token.label_) for token in prompt.ents])
            summary_ner = set([(token.text, token.label_) for token in summary.ents])
        elif "stanza" in str(model):
            prompt_ner = set([(token.text, token.type) for token in prompt.ents])
            summary_ner = set([(token.text, token.type) for token in summary.ents])
        else:
            raise Exception("Model not supported")

        prompt_ner = clean_ners(prompt_ner)
        summary_ner = clean_ners(summary_ner)

        intersecting_ners = prompt_ner.intersection(summary_ner)
        ner_dict = dict(Counter([ner[1] for ner in intersecting_ners]))
        
        if mode == "train":
            return ner_dict
        elif mode == "test":
            return {key: ner_dict.get(key) for key in self.ner_keys}

    
    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary)>0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    
    def run(self, 
            prompts: pd.DataFrame,
            summaries:pd.DataFrame,
            mode:str
        ) -> pd.DataFrame:
        
        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x), 
                skip_special_tokens=True
            )
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x), 
                skip_special_tokens=True
            )

        )
        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence,args=(2,), axis=1 
        )
        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        
        # Crate dataframe with count of each category NERs overlap for all the summaries
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])
    

### get input text ofr training
def get_input_text(row, config):
    sep = " " + config.tokenizer.sep_token + " "   
    text = sep.join(row[config.dataset.input_cols])
    return text


def clean_summary(summary):
    for word in summary.split():
        if word.lower() in contractions:
            summary = summary.replace(word, contractions[word.lower()])
    return summary


def clean_prompts(text):
    # Remove newline characters and extra spaces
    cleaned_text = re.sub(r'\r\n', ' ', text)
    cleaned_text = ' '.join(cleaned_text.split())
    # Remove non-alphanumeric characters except spaces and periods
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.]', '', cleaned_text)
    return cleaned_text


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def get_additional_special_tokens():
    special_tokens_replacement = {
        '\n': '[BR]',
        'sp_token1': '[SUMMARY_START]',
        'sp_token2': '[SUMMARY_END]',
    }
    return special_tokens_replacement


def replace_special_tokens(text):
    special_tokens_replacement = get_additional_special_tokens()
    for key, value in special_tokens_replacement.items():
        text = text.replace(key, value)
    return text


def pad_punctuation(text):
    text = re.sub('([.,!?()-])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    return text


def add_prompt_info(row):
    text = ""
    text += "[PROMPT_QUESTION]" + f" {row['prompt_question']} "
    text += "[PROMPT_TITLE]" + f" {row['prompt_title']} "
    text += "[PROMPT_TEXT]" + f" {row['prompt_text']} "
    text += "[SUMMARY_TEXT]" + f" {row['text']} "
    return text    


def preprocess_text(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    '''
    # template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    # text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # text = re.sub(r"[^a-zA-Z\d]", " ", text) # Remove special Charecters
    # text = re.sub('\n+', '\n', text) 
    # text = re.sub('\.+', '.', text) 
    # text = re.sub(' +', ' ', text) # Remove Extra Spaces 

    # text = resolve_encodings_and_normalize(text)
    text = replace_special_tokens(text)
    return text



def split_prompt_text(config, row):
    text = ""
    sents = row.prompt_text.split(".")
    start = sents[:config.dataset.prompt_text_sent_count]
    if len(start) > 0: text += ". ".join(start)
    return text



def process_prompt_text(row, type="even"):
    text = ""
    sents = row.prompt_text.split(".")
    if type == "even":
        sents = [sent for index, sent in enumerate(sents) if index%2==0 ]
    else:
        sents = [sent for index, sent in enumerate(sents) if index%2!=0 ]

    text += ".".join(sents)
    return text.strip()


def make_folds(df, target_cols, n_splits):
    kfold = GroupKFold(n_splits=n_splits)
    for n, (train_index, val_index) in enumerate(kfold.split(df, df[target_cols], df['prompt_id'])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df


def get_max_len_from_df(df, tokenizer, n_special_tokens=3):
    lengths = []
    tk0 = tqdm(df['input_text'].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_length = max(lengths) + n_special_tokens
    return max_length
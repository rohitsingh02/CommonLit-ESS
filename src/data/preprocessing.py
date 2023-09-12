import codecs
from text_unidecode import unidecode
from typing import Tuple
import re
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from typing import List
from collections import Counter

import nltk
from autocorrect import Speller
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

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
    def __init__(self, model_name: str,) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_name}") 
        self.twd = TreebankWordDetokenizer() 
        self.STOP_WORDS = set(stopwords.words('english'))
        
        self.spacy_ner_model = spacy.load('en_core_web_sm',) 
        self.speller = Speller(lang='en') 
        self.spellchecker = SpellChecker() 
        
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

    def ngram_co_occurrence(self, row, n: int) -> int: # 用于计算原始文本和摘要文本中 n-gram 共现的数量
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)
    
    def ner_overlap_count(self, row, mode:str): # 用于计算两个文本中命名实体（Named Entity）的交集数量。
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

    
    def quotes_count(self, row): # 统计原始文本中出现在摘要文本中的引号的数量。这可以用来衡量原始文本和摘要文本之间的相似性
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary) # 获取摘要文本中提取出所有双引号之间的内容
        if len(quotes_from_summary)>0: # 检查提取的引号内容，检查是否在原始文本中
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text): # 用于检测文本中的拼写错误数量    
        wordlist=text.split()
        # 使用 PySpellChecker 库的 spellchecker 实例，检查单词列表中未知的（即不在词典中的）拼写错误的数量。
        # unknown 方法返回未知的拼写错误单词列表，通过计算其长度来获得错误数量。
        amount_miss = len(list(self.spellchecker.unknown(wordlist))) 

        return amount_miss
    
    def add_spelling_dictionary(self, tokens: List[str]) -> List[str]: # 将一个词汇列表添加到拼写检查的词典中，以便拼写检查器能够更准确地检查文本中的拼写错误。
        """dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens) 
        self.speller.nlp_data.update({token:1000 for token in tokens})
    
    def run(self, 
            prompts: pd.DataFrame,
            summaries:pd.DataFrame,
            mode:str
        ) -> pd.DataFrame:
        
        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(word_tokenize(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(word_tokenize(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )
        
        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(
            lambda x: self.add_spelling_dictionary(x)
        )
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        # fix misspelling
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(
            lambda x: self.speller(x)
        )
        
        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)
        
        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence,args=(2,), axis=1 
        )
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)
        
        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)
        
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])



def clean_summary(summary):
    for word in summary.split():
        if word.lower() in contractions:
            summary = summary.replace(word, contractions[word.lower()])
    # Add space after punctuations
    # clean_summary = summary.replace("\n", "[BR]")
    # Remove HTML tags using BeautifulSoup
    # clean_summary = BeautifulSoup(clean_summary, "html.parser").get_text()
    # Remove special characters and non-printable characters
    # clean_summary = re.sub(r'[^A-Za-z0-9\s]', ' ', clean_summary)
    # Remove extra spaces and newlines
    # clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
    return summary


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
        # 'sp_token1': '[PROMPT_QUESTION]',
        # 'sp_token2': '[PROMPT_TITLE]',
        # 'sp_token3': '[PROMPT_TEXT]',
        # 'sp_token4': '[SUMMARY_TEXT]'
        
        
        # 'Generic_School': '[GENERIC_SCHOOL]',
        # 'Generic_school': '[GENERIC_SCHOOL]',
        # 'SCHOOL_NAME': '[SCHOOL_NAME]',
        # 'STUDENT_NAME': '[STUDENT_NAME]',
        # 'Generic_Name': '[GENERIC_NAME]',
        # 'Genric_Name': '[GENERIC_NAME]',
        # 'Generic_City': '[GENERIC_CITY]',
        # 'LOCATION_NAME': '[LOCATION_NAME]',
        # 'HOTEL_NAME': '[HOTEL_NAME]',
        # 'LANGUAGE_NAME': '[LANGUAGE_NAME]',
        # 'PROPER_NAME': '[PROPER_NAME]',
        # 'OTHER_NAME': '[OTHER_NAME]',
        # 'PROEPR_NAME': '[PROPER_NAME]',
        # 'RESTAURANT_NAME': '[RESTAURANT_NAME]',
        # 'STORE_NAME': '[STORE_NAME]',
        # 'TEACHER_NAME': '[TEACHER_NAME]',
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
    text = clean_summary(text) # newly added
    # text = text.replace('\n', '|')
    text = resolve_encodings_and_normalize(text)
    text = replace_special_tokens(text)
    return text


def make_folds(df, target_cols, n_splits):
    kfold = GroupKFold(n_splits=n_splits)
    for n, (train_index, val_index) in enumerate(kfold.split(df, df[target_cols], df['prompt_id'])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df


def get_max_len_from_df(df, tokenizer, n_special_tokens=3):
    lengths = []
    tk0 = tqdm(df['text'].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_length = max(lengths) + n_special_tokens
    return max_length
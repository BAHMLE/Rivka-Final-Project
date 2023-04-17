#!/usr/bin/env python
# coding: utf-8

# import statements:

import pandas as pd
import boto3
import os
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import tempfile
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau, pearsonr, spearmanr
import joblib
import Bio
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# Read in (cleaned) data

# import data (output from cleaning notebook)
my_region = boto3.session.Session().region_name
# print("region: {}".format(my_region))

DATASET = 'clean2.csv'
MAIN_FOLDER = 's3://tech-x-final-project'
DATA_FOLDER = 'clean-data'


def read_clean_data(ff):
    df = pd.read_csv(os.path.join(MAIN_FOLDER,DATA_FOLDER,ff), header=None)
    df.columns =['protein_sequence','pH','label']
    return df

# Define Vectorization Functions


def vector_seq_len(dd):
    # gets the sequence length for a feature
    dd['n_aa'] = dd['protein_sequence'].map(len)
    return dd
    
def vector_ngram_get_features(a, b, dd):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(a,b))
    term_matrix = vectorizer.fit_transform(dd.protein_sequence)
    features = list(vectorizer.get_feature_names_out())
    return features

def vector_bio(dd):
    # Gets biological vector info
    slst = list(dd.protein_sequence)
    pHlst = list(dd.pH)
    wm, iso, hel, turn, sht, ar, insta, charge, gravy = [], [], [], [],[], [], [], [], []

    for idx, prot in enumerate(slst):
        an = ProteinAnalysis(prot)
        wm.append(an.molecular_weight())
        iso.append(an.isoelectric_point())
        ar.append(an.aromaticity())
        insta.append(an.instability_index())
        charge.append(an.charge_at_pH(pHlst[idx]))
        gravy.append(an.gravy())
        
        stru = an.secondary_structure_fraction()
       
        hel.append(stru[0])
        turn.append(stru[1])
        sht.append(stru[2])
    dd['weight'] = wm
    dd['iso_pt'] = iso
    dd['helix_pct'] = hel
    dd['turn_pct'] = turn
    dd['sheet_pct'] = sht
    dd['aromaticity'] = ar
    dd['instability'] = insta
    dd['charge'] = charge
    dd['gravy'] = gravy

    return dd


# Models


def fit_regression(dd, njobs =7):
    Y = dd.label
    X = dd.drop('label', axis=1)
    reg = LinearRegression(n_jobs=njobs).fit(X, Y)
#     reg.score(X, Y)
    yhat =  reg.predict(X)
    return reg, get_spearman(yhat, Y)[0], reg.score(X,Y)

def use_features(featurelist, dd):
    '''Use a pre-defined list of features and a dataframe with the protein sequences'''
    vectorizer = CountVectorizer(analyzer='char', vocabulary = featurelist)

    term_matrix = vectorizer.fit_transform(dd.protein_sequence)
    features = vectorizer.get_feature_names_out()
    X = term_matrix.toarray()
    features = pd.DataFrame(X, columns = vectorizer.get_feature_names_out())
    return features

def get_spearman(d1, d2):
    coef, p = spearmanr(d1,d2)
    return coef, p


# Function to preprocess data

def pre_process_data(dd, ngram_tup_lst, include_length=True):
    # Run the model on dataframe dd, with ngrams in tuples in a list, and length if 
    # include_length==True
    
    # Get Features (protein sequence)
    features = []
    for ngram in ngram_tup_lst:
        features += vector_ngram_get_features(ngram[0], ngram[1], dd)
    newdf = use_features(features, dd)

    # Get length Features
    dd2 = vector_seq_len(dd)
    newdf['length'] = dd2.n_aa

    
    # Get Protein Features
    dd3 = vector_bio(dd2)
    newdf = pd.concat([newdf, dd3], axis=1)
    newdf = newdf.drop(['protein_sequence','n_aa'],axis=1)
    
    # Scale results
    scaler=StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(newdf), columns = newdf.columns)

    df['label'] = dd.label
    print("columns:\n",df.head().T)
    return df, features, scaler

def prediction_prep(dd, features, scaler):
    # Processing necessary to run a prediction, dd is a DataFrame
    tmp = vector_seq_len(dd)

    tst = use_features(features,tmp)
    tmp2 = vector_bio(dd)
    tst['length'] = tmp.n_aa
    tst['pH'] = tmp.pH
    tst['weight'] = tmp2.weight
    tst['iso_pt'] = tmp2.iso_pt
    tst['helix_pct'] = tmp2.helix_pct
    tst['turn_pct'] = tmp2.turn_pct
    tst['sheet_pct'] = tmp2.sheet_pct
    tst['aromaticity'] = tmp2.aromaticity
    tst['instability'] = tmp2.instability
    tst['charge'] = tmp2.charge
    tst['gravy'] = tmp2.gravy
    
    tst = pd.DataFrame(scaler.fit_transform(tst), columns = tst.columns)
    #.drop('protein_sequence',axis=1)
    return tst
    



def load_itm(name, s3, bucket):
    with tempfile.TemporaryFile() as fp:
        s3.download_fileobj(Fileobj=fp, Bucket=bucket, Key=name)
        fp.seek(0)
        nm = joblib.load(fp)

    return nm








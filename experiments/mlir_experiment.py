import json
import lucene
import numpy as np
import chinese_converter
import ir_measures
from ir_measures import *
from java.nio.file import Paths
from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer
from org.apache.lucene.analysis.ru import RussianAnalyzer
from org.apache.lucene.analysis.fa import PersianAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from sentence_transformers import CrossEncoder
from sklearn.metrics import mean_squared_error 

try:
    lucene.initVM(lucene.CLASSPATH, maxheap='3g')
except Exception as e:
    print(e)

Num_of_Retrieved_Docs = 1000
zho_index_folder = "/path/neuclir1/index_zho"
rus_index_folder = "/path/neuclir1/index_rus"
fas_index_folder = "/path/neuclir1/index_fas"
eng_index_folder =  "/path/neuclir1/index_eng"
dict_clir_id_text = {}
dict_topic_lang_score_normal_rel = {}
dict_qrels_topic_id_doc_id = {}
dict_topic_id_topic_description_eng = {}
dict_topic_id_topic_description_zho = {}
dict_topic_id_topic_description_rus = {}
dict_topic_id_topic_description_fas = {}
dict_user_query = {}
dict_id_english_text = {}

with open('dict_docid_lang.txt') as f: 
    data = f.read() 
dict_docid_lang = json.loads(data)

def calculate_dcg(relevance_scores):
    return np.sum([((2 ** rel) - 1) / np.log2(rank + 2) for rank, rel in enumerate(relevance_scores)])

def min_max_normalization(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

def get_ratios_doclist(dict_docid_lang, doclist):
    count_fas = 0
    count_rus = 0
    count_zho = 0
    for docid in doclist: 
        if dict_docid_lang[docid] == 'fas': 
            count_fas = count_fas + 1
        elif dict_docid_lang[docid] == 'rus': 
            count_rus = count_rus + 1
        elif dict_docid_lang[docid] == 'zho': 
            count_zho = count_zho + 1
    total = count_fas + count_rus + count_zho
    return count_fas/total, count_rus/total, count_zho/total

def get_ratios_run_file(dict_docid_lang, run_file, k=0):
    ratios_fas = []
    ratios_rus = []
    ratios_zho = []
    dict_qid_doclist = {}
    with open(run_file) as f: 
        for line in f:
            data = line.split()
            qid = data[0]
            if qid not in dict_qid_doclist:
                dict_qid_doclist[qid] = []
            dict_qid_doclist[qid].append(data[2])
    for qid in dict_qid_doclist:
        doclist = dict_qid_doclist[qid]
        if k == 0:
            k = len(doclist)
        doclist = doclist[:k]
        r_fas, r_rus, r_zho = get_ratios_doclist(dict_docid_lang, doclist)
        ratios_fas.append(r_fas)
        ratios_rus.append(r_rus)
        ratios_zho.append(r_zho)
    return sum(ratios_fas)/len(ratios_fas), sum(ratios_rus)/len(ratios_rus), sum(ratios_zho)/len(ratios_zho)        

def alpha_dcg(relevance, ranking, alpha=0.5):
    """
    Compute the alpha-DCG for a list of ranked documents and their relevance scores.
    Parameters:
    - relevance: Dictionary where keys are document IDs and values are lists of relevance scores for subtopics.
    - ranking: List of document IDs representing the ranked list.
    - alpha: The penalty parameter for redundancy.
    Returns:
    - The alpha-DCG score.
    """
    dcg = 0.0
    topic_coverage = {}
    for rank, doc_id in enumerate(ranking):
        if doc_id not in relevance:
            continue
        scores = relevance[doc_id]
        gain_sum = 0.0
        for topic_idx, score in enumerate(scores):
            if score > 0:
                if topic_idx not in topic_coverage:
                    topic_coverage[topic_idx] = 0
                gain = score * (1 - alpha) ** topic_coverage[topic_idx]
                gain_sum += gain
                topic_coverage[topic_idx] += 1
        dcg += gain_sum / np.log2(rank + 2)
    return dcg

def ideal_alpha_dcg(relevance, alpha=0.5):
    """
    Compute the ideal alpha-DCG (IDCG) for a list of relevance scores.
    Parameters:
    - relevance: Dictionary where keys are document IDs and values are lists of relevance scores for subtopics.
    - alpha: The penalty parameter for redundancy.
    Returns:
    - The ideal alpha-DCG score.
    """
    # Sort documents by their maximum possible gain in descending order
    sorted_docs = sorted(relevance.keys(), key=lambda doc_id: sum(relevance[doc_id]), reverse=True)
    return alpha_dcg(relevance, sorted_docs, alpha)

def alpha_ndcg(relevance, ranking, alpha=0.5):
    """
    Compute the alpha-NDCG for a list of ranked documents and their relevance scores.
    Parameters:
    - relevance: Dictionary where keys are document IDs and values are lists of relevance scores for subtopics.
    - ranking: List of document IDs representing the ranked list.
    - alpha: The penalty parameter for redundancy.
    Returns:
    - The alpha-NDCG score.
    """
    dcg = alpha_dcg(relevance, ranking, alpha)
    idcg = ideal_alpha_dcg(relevance, alpha)
    return dcg / idcg if idcg > 0 else 0.0

def get_doc_rel_lang(dict_docid_lang, dict_qrels_topic_id_doc_id, qid, docid):
    rel_fas_rus_zho = [0, 0, 0]
    rel = 0
    if docid in dict_qrels_topic_id_doc_id[qid]: 
        rel = dict_qrels_topic_id_doc_id[qid][docid]
    if dict_docid_lang[docid] == 'fas':
        rel_fas_rus_zho[0] = rel
    elif dict_docid_lang[docid] == 'rus':
        rel_fas_rus_zho[1] = rel
    elif dict_docid_lang[docid] == 'zho':
        rel_fas_rus_zho[2] = rel
    return rel_fas_rus_zho    

def get_alpha_ndcg_run_file(dict_docid_lang, dict_qrels_topic_id_doc_id, run_file, alpha, k=0):
    alpha_ndcgs_list = []
    dict_qid_doclist = {}
    with open(run_file) as f: 
        for line in f:
            data = line.split()
            qid = data[0]
            if qid not in dict_qid_doclist:
                dict_qid_doclist[qid] = []
            dict_qid_doclist[qid].append(data[2])
    for qid in dict_qid_doclist:
        doclist = dict_qid_doclist[qid]
        if k == 0:
            k = len(doclist)
        doclist = doclist[:k]
        relevance = {}
        for docid in doclist:
            relevance[docid] = get_doc_rel_lang(dict_docid_lang, dict_qrels_topic_id_doc_id, qid, docid)
        alpha_ndcgs_list.append(alpha_ndcg(relevance, doclist, alpha))
    return sum(alpha_ndcgs_list)/len(alpha_ndcgs_list)

topics = [t for t in map(json.loads, open("neuclir-2023-topics.0605.jsonl"))]
for t in topics:
    for topic in t['topics']: 
        if topic['lang'] == 'eng':
            dict_topic_id_topic_description_eng[t['topic_id']] = topic['topic_description']
        elif topic['lang'] == 'fas' and topic['source'] == 'google translation':
            dict_topic_id_topic_description_fas[t['topic_id']] = topic['topic_description']
        elif topic['lang'] == 'rus' and topic['source'] == 'google translation':
            dict_topic_id_topic_description_rus[t['topic_id']] = topic['topic_description']
        elif topic['lang'] == 'zho' and topic['source'] == 'google translation':
            dict_topic_id_topic_description_zho[t['topic_id']] = topic['topic_description']

with open("qrels.final", "r") as fp:
    lines = fp.readlines()
    for line in lines: 
        cols = line.strip().split()
        topic_id = cols[0]
        doc_id = cols[2]
        qrels = int(cols[3])
        if topic_id in dict_qrels_topic_id_doc_id: 
            dict_qrels_topic_id_doc_id[topic_id][doc_id] = qrels
        else: 
            dict_qrels_topic_id_doc_id[topic_id] = {}
            dict_qrels_topic_id_doc_id[topic_id][doc_id] = qrels                

for t in topics:
    user_query = {}
    for topic in t['topics']: 
        if topic['lang'] == 'eng': 
            user_query[topic['lang']] = topic['topic_title']
        else: 
            if topic['source'] == 'google translation':
                if topic['lang'] == 'zho':
                    user_query[topic['lang']] = chinese_converter.to_simplified(topic['topic_title'])
                    #user_query[topic['lang']] = topic['topic_title']
                else:
                    user_query[topic['lang']] = topic['topic_title']
    dict_user_query[t['topic_id']] = user_query

smart_chinese_analyzer = SmartChineseAnalyzer()
russian_analyzer = RussianAnalyzer()
persian_analyzer = PersianAnalyzer()
english_analyzer = EnglishAnalyzer()
whitespace_analyzer = WhitespaceAnalyzer()

# Open the Lucene indexes
index_directory_zho = FSDirectory.open(Paths.get(zho_index_folder))
index_directory_rus = FSDirectory.open(Paths.get(rus_index_folder))
index_directory_fas = FSDirectory.open(Paths.get(fas_index_folder))
index_directory_eng = FSDirectory.open(Paths.get(eng_index_folder))
# Use IndexSearcher to search the index
index_searcher_zho = IndexSearcher(DirectoryReader.open(index_directory_zho))
index_searcher_zho.setSimilarity(BM25Similarity())
index_searcher_rus = IndexSearcher(DirectoryReader.open(index_directory_rus))
index_searcher_rus.setSimilarity(BM25Similarity())
index_searcher_fas = IndexSearcher(DirectoryReader.open(index_directory_fas))
index_searcher_fas.setSimilarity(BM25Similarity())
index_searcher_eng = IndexSearcher(DirectoryReader.open(index_directory_eng))
index_searcher_eng.setSimilarity(BM25Similarity())
# Create a QueryParser with the same analyzer used during indexing
query_parser_zho = QueryParser("titletext", smart_chinese_analyzer)
query_parser_rus = QueryParser("titletext", russian_analyzer)
query_parser_fas = QueryParser("titletext", persian_analyzer)
query_parser_eng = QueryParser("titletext", english_analyzer)
query_parser_id = QueryParser("id", whitespace_analyzer)

zho_docs_num = index_searcher_zho.getIndexReader().numDocs();
rus_docs_num = index_searcher_rus.getIndexReader().numDocs();
fas_docs_num = index_searcher_fas.getIndexReader().numDocs();
eng_docs_num = index_searcher_eng.getIndexReader().numDocs();
docs_total_three = zho_docs_num + rus_docs_num + fas_docs_num
zho_docs_ratio = zho_docs_num / docs_total_three
rus_docs_ratio = rus_docs_num / docs_total_three
fas_docs_ratio = fas_docs_num / docs_total_three

Num_of_Retrieved_Docs_zho = int(zho_docs_ratio*Num_of_Retrieved_Docs)
Num_of_Retrieved_Docs_rus = int(rus_docs_ratio*Num_of_Retrieved_Docs)
Num_of_Retrieved_Docs_fas = Num_of_Retrieved_Docs - Num_of_Retrieved_Docs_zho - Num_of_Retrieved_Docs_rus

for key in dict_user_query: 
    query_topic_id = key
    user_query = dict_user_query[key]
    # Parse the user's query
    if 'zho' not in user_query:
        continue
    # Remove * because it crashes SmartChineseAnalyzer.
    parsed_query_zho = query_parser_zho.parse(user_query['zho'].replace('*', ' '))    
    search_results_zho = index_searcher_zho.search(parsed_query_zho, Num_of_Retrieved_Docs_zho)
    results_scores_zho = []
    results_scores_zho_id = []
    dict_id_score = {}
    # Process search results
    for score_doc in search_results_zho.scoreDocs:
        results_scores_zho.append(score_doc.score)
        doc_id = score_doc.doc
        #print(doc_id)
        doc = index_searcher_zho.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        text = doc.get("titletext")
        #url = doc.get("url")
        dict_clir_id_text[id] = text
        results_scores_zho_id.append(id)
        dict_id_score[id] = score_doc.score
    ranked_list = results_scores_zho_id
    #print(ranked_list)
    relevance_scores = []
    for i in range(len(ranked_list)):
        doc_id = ranked_list[i]
        if doc_id in dict_qrels_topic_id_doc_id[query_topic_id]: 
            relevance_scores.append(dict_qrels_topic_id_doc_id[query_topic_id][doc_id])
        else: 
            relevance_scores.append(0)
    if key not in dict_topic_lang_score_normal_rel:
        dict_topic_lang_score_normal_rel[key] = {}
    dict_topic_lang_score_normal_rel[key]['zho'] = []
    dict_topic_lang_score_normal_rel[key]['zho'].append(list(zip(results_scores_zho, relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['zho'].append(list(zip(min_max_normalization(results_scores_zho), relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['zho'].append(ranked_list)
    dcg = calculate_dcg(relevance_scores)
    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance_scores)
    # Calculate NDCG score
    ndcg = dcg / idcg
    # print("Discounted Cumulative Gain (DCG):", dcg)
    # print("Ideal Discounted Cumulative Gain (IDCG):", idcg)
    # print("NDCG score:", ndcg)    

for key in dict_user_query: 
    query_topic_id = key
    user_query = dict_user_query[key]
    # Parse the user's query
    if 'rus' not in user_query:
        continue
    #print(key, user_query['rus'])
    parsed_query_rus = query_parser_rus.parse(user_query['rus'])    
    search_results_rus = index_searcher_rus.search(parsed_query_rus, Num_of_Retrieved_Docs_rus)
    results_scores_rus = []
    results_scores_rus_id = []
    dict_id_score = {}
    # Process search results
    for score_doc in search_results_rus.scoreDocs:
        results_scores_rus.append(score_doc.score)
        doc_id = score_doc.doc
        #print(doc_id)
        doc = index_searcher_rus.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        text = doc.get("titletext")
        #url = doc.get("url")
        dict_clir_id_text[id] = text
        results_scores_rus_id.append(id)
        dict_id_score[id] = score_doc.score
    ranked_list = results_scores_rus_id
    #print(ranked_list)
    relevance_scores = []
    for i in range(len(ranked_list)):
        doc_id = ranked_list[i]
        if doc_id in dict_qrels_topic_id_doc_id[query_topic_id]: 
            relevance_scores.append(dict_qrels_topic_id_doc_id[query_topic_id][doc_id])
        else: 
            relevance_scores.append(0)
    if key not in dict_topic_lang_score_normal_rel:
        dict_topic_lang_score_normal_rel[key] = {}
    dict_topic_lang_score_normal_rel[key]['rus'] = []
    dict_topic_lang_score_normal_rel[key]['rus'].append(list(zip(results_scores_rus, relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['rus'].append(list(zip(min_max_normalization(results_scores_rus), relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['rus'].append(ranked_list)
    dcg = calculate_dcg(relevance_scores)
    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance_scores)
    # Calculate NDCG score
    ndcg = dcg / idcg
    # print("Discounted Cumulative Gain (DCG):", dcg)
    # print("Ideal Discounted Cumulative Gain (IDCG):", idcg)
    # print("NDCG score:", ndcg)

for key in dict_user_query: 
    query_topic_id = key
    user_query = dict_user_query[key]
    # Parse the user's query
    if 'fas' not in user_query:
        continue
    #print(key, user_query['fas'])
    parsed_query_fas = query_parser_fas.parse(user_query['fas'])    
    search_results_fas = index_searcher_fas.search(parsed_query_fas, Num_of_Retrieved_Docs_fas)
    results_scores_fas = []
    results_scores_fas_id = []
    dict_id_score = {}
    # Process search results
    for score_doc in search_results_fas.scoreDocs:
        results_scores_fas.append(score_doc.score)
        doc_id = score_doc.doc
        #print(doc_id)
        doc = index_searcher_fas.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        text = doc.get("titletext")
        #url = doc.get("url")
        dict_clir_id_text[id] = text
        results_scores_fas_id.append(id)
        dict_id_score[id] = score_doc.score
    ranked_list = results_scores_fas_id
    #print(ranked_list)
    relevance_scores = []
    for i in range(len(ranked_list)):
        doc_id = ranked_list[i]
        if doc_id in dict_qrels_topic_id_doc_id[query_topic_id]: 
            relevance_scores.append(dict_qrels_topic_id_doc_id[query_topic_id][doc_id])
        else: 
            relevance_scores.append(0)
    if key not in dict_topic_lang_score_normal_rel:
        dict_topic_lang_score_normal_rel[key] = {}
    dict_topic_lang_score_normal_rel[key]['fas'] = []
    dict_topic_lang_score_normal_rel[key]['fas'].append(list(zip(results_scores_fas, relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['fas'].append(list(zip(min_max_normalization(results_scores_fas), relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['fas'].append(ranked_list)
    dcg = calculate_dcg(relevance_scores)
    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance_scores)
    # Calculate NDCG score
    ndcg = dcg / idcg
    # print("Discounted Cumulative Gain (DCG):", dcg)
    # print("Ideal Discounted Cumulative Gain (IDCG):", idcg)
    # print("NDCG score:", ndcg)

for key in dict_user_query: 
    query_topic_id = key
    user_query = dict_user_query[key]
    # Parse the user's query
    if 'eng' not in user_query:
        continue
    #print(key, user_query['eng'])
    parsed_query_eng = query_parser_eng.parse(user_query['eng'])    
    #parsed_query_eng = query_parser_eng.parse(dict_topic_id_topic_description_eng[key].replace('/', ' ') )
    search_results_eng = index_searcher_eng.search(parsed_query_eng, Num_of_Retrieved_Docs)
    results_scores_eng = []
    results_scores_eng_id = []
    dict_id_score = {}
    # Process search results
    for score_doc in search_results_eng.scoreDocs:
        results_scores_eng.append(score_doc.score)
        doc_id = score_doc.doc
        #print(doc_id)
        doc = index_searcher_eng.doc(doc_id)
        title = doc.get("title")
        titletext = doc.get("titletext")
        url = doc.get("url")
        id = doc.get("id")
        dict_id_english_text[id] = titletext
        results_scores_eng_id.append(id)
        dict_id_score[id] = score_doc.score
    ranked_list = results_scores_eng_id
    #print(ranked_list)
    relevance_scores = []
    for i in range(len(ranked_list)):
        doc_id = ranked_list[i]
        if doc_id in dict_qrels_topic_id_doc_id[query_topic_id]: 
            relevance_scores.append(dict_qrels_topic_id_doc_id[query_topic_id][doc_id])
        else: 
            relevance_scores.append(0)
    if key not in dict_topic_lang_score_normal_rel:
        dict_topic_lang_score_normal_rel[key] = {}
    dict_topic_lang_score_normal_rel[key]['eng'] = []
    dict_topic_lang_score_normal_rel[key]['eng'].append(list(zip(results_scores_eng, relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['eng'].append(list(zip(min_max_normalization(results_scores_eng), relevance_scores)))
    dict_topic_lang_score_normal_rel[key]['eng'].append(ranked_list)
    dcg = calculate_dcg(relevance_scores)
    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance_scores)
    # Calculate NDCG score
    ndcg = dcg / idcg

# useNormalizedScore = 0 to use original score from Lucene
def merge_lang_ranked_list(topic_id, dict_topic_lang_score_normal_rel, useNormalizedScore=1):
    ranked_docs_socre = []
    for lang in dict_topic_lang_score_normal_rel[topic_id]:
        if lang != 'eng':
            ranked_docs_socre = ranked_docs_socre + [(y, x[0]) for x, y in zip(dict_topic_lang_score_normal_rel[topic_id][lang][useNormalizedScore], dict_topic_lang_score_normal_rel[topic_id][lang][2])]
    return sorted(ranked_docs_socre, key=lambda x: x[1], reverse=True)    

def get_translated_title_text_by_id(doc_id, index_searcher_eng, query_parser_id): 
    query_id = query_parser_id.parse(doc_id)
    #print(query_id)
    search_results = index_searcher_eng.search(query_id, 1)
    str_to_get = ''
    for score_doc in search_results.scoreDocs:
        tmp_id = score_doc.doc
        doc = index_searcher_eng.doc(tmp_id)
        str_to_get = str_to_get + doc.get("titletext") + ' '
    return str_to_get

with open('mlir-teamli-ratio_run1-fair', 'w') as f: 
    for topic_id in dict_topic_lang_score_normal_rel:
        ranked_docs_score = merge_lang_ranked_list(topic_id, dict_topic_lang_score_normal_rel)
        for i in range(Num_of_Retrieved_Docs):
                print(topic_id, 'Q0', ranked_docs_score[i][0], i+1, ranked_docs_score[i][1], 'mlir-teamli-1ANCPRS_run1-fair', file=f)    

with open('mlir-teamli-ratio_run2-fair', 'w') as f: 
    for topic_id in dict_topic_lang_score_normal_rel:
        ranked_docs_score = merge_lang_ranked_list(topic_id, dict_topic_lang_score_normal_rel, useNormalizedScore=0)
        for i in range(Num_of_Retrieved_Docs):
                print(topic_id, 'Q0', ranked_docs_score[i][0], i+1, ranked_docs_score[i][1], 'mlir-teamli-1ANCPRS_run2-fair', file=f)   

qrels = ir_measures.read_trec_qrels('qrels.final.gains.mlir')
run = ir_measures.read_trec_run('mlir-teamli-ratio_run1-fair')
print('mlir-teamli-ratio_run1-fair', ir_measures.calc_aggregate([nDCG@20, nDCG@50, R@20, R@50, R@100, R@1000], qrels, run)
print('mlir-teamli-ratio_run1-fair alpha-nDCG', get_alpha_ndcg_run_file(dict_docid_lang, dict_qrels_topic_id_doc_id, 'mlir-teamli-ratio_run1-fair', 0.5, k=20)))
print('mlir-teamli-ratio_run1-fair ratios fas:rus:zho', get_ratios_run_file(dict_docid_lang, 'mlir-teamli-ratio_run1-fair', 20))

qrels = ir_measures.read_trec_qrels('qrels.final.gains.mlir')
run = ir_measures.read_trec_run('mlir-teamli-ratio_run2-fair')
print('mlir-teamli-ratio_run2-fair', ir_measures.calc_aggregate([nDCG@20, nDCG@50, R@20, R@50, R@100, R@1000], qrels, run)
print('mlir-teamli-ratio_run2-fair alpha-nDCG', get_alpha_ndcg_run_file(dict_docid_lang, dict_qrels_topic_id_doc_id, 'mlir-teamli-ratio_run2-fair', 0.5, k=20)))
print('mlir-teamli-ratio_run2-fair ratios fas:rus:zho', get_ratios_run_file(dict_docid_lang, 'mlir-teamli-ratio_run2-fair', 20))

dict_topic_title_english_doc_score1 = {}
model1 = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=200)
for topic_id in dict_topic_lang_score_normal_rel:
    doc_id_list = dict_topic_lang_score_normal_rel[topic_id]['fas'][2]+dict_topic_lang_score_normal_rel[topic_id]['rus'][2]+dict_topic_lang_score_normal_rel[topic_id]['zho'][2]
    query = dict_user_query[topic_id]['eng']+ ' ' +dict_topic_id_topic_description_eng[topic_id]
    #query = dict_topic_id_topic_description_eng[topic_id]
    english_docs_list = []
    query_list = []
    for i in range(len(doc_id_list)):
        tmp_id = doc_id_list[i]
        english_docs_list.append(get_translated_title_text_by_id(tmp_id, index_searcher_eng, query_parser_id))
        query_list.append(query)
    scores = model1.predict(list(zip(query_list, english_docs_list)))
    dict_topic_title_english_doc_score1[topic_id] = sorted(zip(doc_id_list, scores), key = lambda x : x[1], reverse = True)

with open('clir-test_run_TinyBERT1000fairratio', 'w') as f: 
    for topic_id in dict_topic_title_english_doc_score1:
        ranked_docs_socres = dict_topic_title_english_doc_score1[topic_id]
        for i in range(len(ranked_docs_socres)):
            print(topic_id, 'Q0', ranked_docs_socres[i][0], i+1, ranked_docs_socres[i][1], 'clir-test_run_TinyBERT1000fairratio', file=f) 

qrels = ir_measures.read_trec_qrels('qrels.final.gains.mlir')
run = ir_measures.read_trec_run('clir-test_run_TinyBERT1000fairratio')
print('clir-test_run_TinyBERT1000fairratio', ir_measures.calc_aggregate([nDCG@20, nDCG@50, R@20, R@50, R@100, R@1000], qrels, run))
print('clir-test_run_TinyBERT1000fairratio alpha-nDCG', get_alpha_ndcg_run_file(dict_docid_lang, dict_qrels_topic_id_doc_id, 'clir-test_run_TinyBERT1000fairratio', 0.5, k=20))
print('clir-test_run_TinyBERT1000fairratio ratios fas:rus:zho', get_ratios_run_file(dict_docid_lang, 'clir-test_run_TinyBERT1000fairratio', 20))

dict_topic_title_english_doc_score1 = {}
model1 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=200)
for topic_id in dict_topic_lang_score_normal_rel:
    doc_id_list = dict_topic_lang_score_normal_rel[topic_id]['fas'][2]+dict_topic_lang_score_normal_rel[topic_id]['rus'][2]+dict_topic_lang_score_normal_rel[topic_id]['zho'][2]
    query = dict_user_query[topic_id]['eng']+ ' ' +dict_topic_id_topic_description_eng[topic_id]
    #query = dict_topic_id_topic_description_eng[topic_id]
    english_docs_list = []
    query_list = []
    for i in range(len(doc_id_list)):
        tmp_id = doc_id_list[i]
        english_docs_list.append(get_translated_title_text_by_id(tmp_id, index_searcher_eng, query_parser_id))
        query_list.append(query)
    scores = model1.predict(list(zip(query_list, english_docs_list)))
    dict_topic_title_english_doc_score1[topic_id] = sorted(zip(doc_id_list, scores), key = lambda x : x[1], reverse = True)

with open('clir-test_run_MiniLM-L-61000fairratio', 'w') as f: 
    for topic_id in dict_topic_title_english_doc_score1:
        ranked_docs_socres = dict_topic_title_english_doc_score1[topic_id]
        for i in range(len(ranked_docs_socres)):
            print(topic_id, 'Q0', ranked_docs_socres[i][0], i+1, ranked_docs_socres[i][1], 'clir-test_run_MiniLM-L-61000fairratio', file=f) 

qrels = ir_measures.read_trec_qrels('qrels.final.gains.mlir')
run = ir_measures.read_trec_run('clir-test_run_MiniLM-L-61000fairratio')
print('clir-test_run_MiniLM-L-61000fairratio', ir_measures.calc_aggregate([nDCG@20, nDCG@50, R@20, R@50, R@100, R@1000], qrels, run))
print('clir-test_run_MiniLM-L-61000fairratio alpha-nDCG', get_alpha_ndcg_run_file(dict_docid_lang, dict_qrels_topic_id_doc_id, 'clir-test_run_MiniLM-L-61000fairratio', 0.5, k=20))
print('clir-test_run_MiniLM-L-61000fairratio ratios fas:rus:zho', get_ratios_run_file(dict_docid_lang, 'clir-test_run_MiniLM-L-61000fairratio', 20))

# Close the Lucene index readers
index_searcher_zho.getIndexReader().close()
index_searcher_rus.getIndexReader().close()
index_searcher_fas.getIndexReader().close()
index_searcher_eng.getIndexReader().close()

# Close the Lucene index directories
index_directory_zho.close()
index_directory_rus.close()
index_directory_fas.close()
index_directory_eng.close()

relevant_doc_ratio  = [0.22233963370804066, 0.4609672222726932, 0.3166931440192661]
a = [0.19039473684210528, 0.4227631578947369, 0.3868421052631579]
print(mean_squared_error(relevant_doc_ratio, a)*100)

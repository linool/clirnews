import os
import lucene
import translators as ts
from java.nio.file import Paths
from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer
from org.apache.lucene.analysis.ru import RussianAnalyzer
from org.apache.lucene.analysis.fa import PersianAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, ScoreDoc
from org.apache.lucene.search.similarities import BM25Similarity
from sentence_transformers import CrossEncoder
from flask import Flask, render_template, request, send_from_directory

try:
    lucene.initVM(lucene.CLASSPATH, maxheap='3g')
except Exception as e:
    print(e)

app = Flask(__name__)

# Number of documents to return for each query
Num_of_Retrieved_Docs = 50
Cache = {}
fas_index_folder = "/path/index_fas"
rus_index_folder = "/path/index_rus"
zho_index_folder = "/path/index_zho"
eng_index_folder = "/path/index_eng"
model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=200)
#model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2", max_length=200)
#model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=200)

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

def search_by_query(input_query):
    user_query = {}
    dict_clir_id_title = {}
    dict_clir_id_text = {}
    dict_id_english_title = {}
    dict_id_english_titletext = {}
    dict_id_url = {}
    #dict_id_lucene_score = {}
    user_query['zho'] = ts.translate_text(input_query, 'google', 'auto', 'zh')
    user_query['rus'] = ts.translate_text(input_query, 'google', 'auto', 'ru')
    user_query['fas'] = ts.translate_text(input_query, 'google', 'auto', 'fa')
    user_query['eng'] = ts.translate_text(input_query, 'google', 'auto', 'en')  
    parsed_query_zho = query_parser_zho.parse(user_query['zho'].replace('*', ' '))    
    search_results_zho = index_searcher_zho.search(parsed_query_zho, Num_of_Retrieved_Docs_zho)
    # Process Chinese search results
    for score_doc in search_results_zho.scoreDocs:
        doc_id = score_doc.doc
        doc = index_searcher_zho.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        titletext = doc.get("titletext")
        index_slash_n = titletext.find(' /n ')
        url = doc.get("url")
        dict_clir_id_title[id] = titletext[:index_slash_n]
        dict_clir_id_text[id] = titletext[index_slash_n+4:]
        dict_id_url[id] = url
        #dict_id_lucene_score[id] = score_doc.score
    parsed_query_rus = query_parser_rus.parse(user_query['rus'])    
    search_results_rus = index_searcher_rus.search(parsed_query_rus, Num_of_Retrieved_Docs_rus)
    # Process Russian search results
    for score_doc in search_results_rus.scoreDocs:
        doc_id = score_doc.doc
        doc = index_searcher_rus.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        titletext = doc.get("titletext")
        index_slash_n = titletext.find(' /n ')
        url = doc.get("url")
        dict_clir_id_title[id] = titletext[:index_slash_n]
        dict_clir_id_text[id] = titletext[index_slash_n+4:]
        dict_id_url[id] = url
        #dict_id_lucene_score[id] = score_doc.score
    parsed_query_fas = query_parser_fas.parse(user_query['fas'])    
    search_results_fas = index_searcher_fas.search(parsed_query_fas, Num_of_Retrieved_Docs_fas)
    # Process Persian search results
    for score_doc in search_results_fas.scoreDocs:
        doc_id = score_doc.doc
        doc = index_searcher_fas.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        titletext = doc.get("titletext")
        index_slash_n = titletext.find(' /n ')
        url = doc.get("url")
        dict_clir_id_title[id] = titletext[:index_slash_n]
        dict_clir_id_text[id] = titletext[index_slash_n+4:]
        dict_id_url[id] = url
        #dict_id_lucene_score[id] = score_doc.score
    for id in dict_clir_id_title:
        titletext = get_translated_title_text_by_id(id, index_searcher_eng, query_parser_id)
        dict_id_english_titletext[id] = titletext
        dict_id_english_title[id] = get_english_title(titletext)
    doc_id_list = []
    query_list = []
    english_docs_list = []
    for id in dict_id_english_titletext:
        doc_id_list.append(id)
        query_list.append(user_query['eng'])
        english_docs_list.append(dict_id_english_titletext[id])
    scores = model.predict(list(zip(query_list, english_docs_list)))
    sorted_id_scores = sorted(zip(doc_id_list, scores), key = lambda x : x[1], reverse = True)
    news = []
    for id_score in sorted_id_scores: 
        id = id_score[0]
        dict_news = {}
        dict_news['id'] = id
        dict_news['title'] = dict_clir_id_title[id]
        dict_news['title_english'] = dict_id_english_title[id]
        dict_news['text'] = dict_clir_id_text[id][:600] + '...'
        dict_news['text_english'] = get_english_text(dict_id_english_titletext[id])
        dict_news['url'] = dict_id_url[id]
        news.append(dict_news)
    return news

def basic_search_by_query(input_query):
    user_query = {}
    dict_clir_id_title = {}
    dict_clir_id_text = {}
    dict_id_english_title = {}
    dict_id_english_titletext = {}
    dict_id_url = {}
    dict_id_lucene_score = {}
    user_query['zho'] = ts.translate_text(input_query, 'google', 'auto', 'zh')
    user_query['rus'] = ts.translate_text(input_query, 'google', 'auto', 'ru')
    user_query['fas'] = ts.translate_text(input_query, 'google', 'auto', 'fa')
    user_query['eng'] = ts.translate_text(input_query, 'google', 'auto', 'en')
    parsed_query_zho = query_parser_zho.parse(user_query['zho'].replace('*', ' '))
    search_results_zho = index_searcher_zho.search(parsed_query_zho, Num_of_Retrieved_Docs_zho)
    # Process Chinese search results
    for score_doc in search_results_zho.scoreDocs:
        doc_id = score_doc.doc
        doc = index_searcher_zho.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        titletext = doc.get("titletext")
        index_slash_n = titletext.find(' /n ')
        url = doc.get("url")
        dict_clir_id_title[id] = titletext[:index_slash_n]
        dict_clir_id_text[id] = titletext[index_slash_n+4:]
        dict_id_url[id] = url
        dict_id_lucene_score[id] = score_doc.score
    parsed_query_rus = query_parser_rus.parse(user_query['rus'])
    search_results_rus = index_searcher_rus.search(parsed_query_rus, Num_of_Retrieved_Docs_rus)
    # Process Russian search results
    for score_doc in search_results_rus.scoreDocs:
        doc_id = score_doc.doc
        doc = index_searcher_rus.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        titletext = doc.get("titletext")
        index_slash_n = titletext.find(' /n ')
        url = doc.get("url")
        dict_clir_id_title[id] = titletext[:index_slash_n]
        dict_clir_id_text[id] = titletext[index_slash_n+4:]
        dict_id_url[id] = url
        dict_id_lucene_score[id] = score_doc.score
    parsed_query_fas = query_parser_fas.parse(user_query['fas'])
    search_results_fas = index_searcher_fas.search(parsed_query_fas, Num_of_Retrieved_Docs_fas)
    # Process Persian search results
    for score_doc in search_results_fas.scoreDocs:
        doc_id = score_doc.doc
        doc = index_searcher_fas.doc(doc_id)
        id = doc.get("id")
        #title = doc.get("title")
        titletext = doc.get("titletext")
        index_slash_n = titletext.find(' /n ')
        url = doc.get("url")
        dict_clir_id_title[id] = titletext[:index_slash_n]
        dict_clir_id_text[id] = titletext[index_slash_n+4:]
        dict_id_url[id] = url
        dict_id_lucene_score[id] = score_doc.score
    for id in dict_clir_id_title:
        titletext = get_translated_title_text_by_id(id, index_searcher_eng, query_parser_id)
        dict_id_english_titletext[id] = titletext
        dict_id_english_title[id] = get_english_title(titletext)
    sorted_ids = sorted(dict_id_lucene_score, reverse = True)
    news = []
    for id in sorted_ids:
        dict_news = {}
        dict_news['id'] = id
        dict_news['title'] = dict_clir_id_title[id]
        dict_news['title_english'] = dict_id_english_title[id]
        dict_news['text'] = dict_clir_id_text[id][:600] + '...'
        dict_news['text_english'] = get_english_text(dict_id_english_titletext[id])
        dict_news['url'] = dict_id_url[id]
        news.append(dict_news)
    return news
    
def get_translated_title_text_by_id(id, index_searcher_eng, query_parser_id): 
    query_id = query_parser_id.parse(id)
    search_results = index_searcher_eng.search(query_id, 1)
    for score_doc in search_results.scoreDocs:
        tmp_id = score_doc.doc
        doc = index_searcher_eng.doc(tmp_id)
        return doc.get("titletext")
    return ''

def get_english_title(titletext):
    index_backslash_n = titletext.find(' \n ')
    return titletext[:index_backslash_n]

def get_english_text(titletext):
    index_backslash_n = titletext.find(' \n ')
    return titletext[index_backslash_n+3:]

@app.route("/", methods=["GET","POST"])
def search():
    if request.method == "POST":
        lucene.getVMEnv().attachCurrentThread()
        input_query = request.form["query"]
        input_query = input_query.strip()
        input_query = input_query.replace('/',' ')
        if input_query in Cache:
            results = Cache[input_query]
        else:
            results = search_by_query(input_query)
            Cache[input_query] = results
        return render_template("results.html", news=results)
    else:
        lucene.getVMEnv().attachCurrentThread()
        return render_template("search.html")

@app.route("/basic", methods=["GET","POST"])
def basic_search():
    if request.method == "POST":
        lucene.getVMEnv().attachCurrentThread()
        input_query = request.form["query"]
        input_query = input_query.strip()
        input_query = input_query.replace('/',' ')
        if input_query in Cache:
            results = Cache[input_query]
        else:
            results = basic_search_by_query(input_query)
            #Cache[input_query] = results
        return render_template("results.html", news=results)
    else:
        lucene.getVMEnv().attachCurrentThread()
        return render_template("search.html")

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)

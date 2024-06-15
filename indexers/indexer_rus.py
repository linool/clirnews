import lucene
import os
import json
from java.nio.file import Paths
from org.apache.lucene.analysis.ru import RussianAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexOptions, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory

try:
    lucene.initVM(lucene.CLASSPATH, maxheap='3g')
except Exception as e:
    print(e)

def create_document(json_data):
    t1 = FieldType()
    t1.setStored(True)
    t1.setTokenized(False)
    t1.setIndexOptions(IndexOptions.DOCS)  
    t2 = FieldType()
    t2.setStored(True)
    t2.setTokenized(True)
    t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    doc = Document()
    doc.add(Field('id', json_data['id'], t1))
    doc.add(Field('url', json_data['url'], t1))
    doc.add(Field('title', json_data['title'], t2))
    doc.add(Field('titletext', json_data['title']+' \n '+json_data['text'], t2))
    return doc

docs_folder = '/path/neuclir1/rus'
index_directory = FSDirectory.open(Paths.get("/path/index_rus"))
config = IndexWriterConfig(RussianAnalyzer())
writer = IndexWriter(index_directory, config)
for root, dirs, files in os.walk(docs_folder):
    for name in files:
        file_path = os.path.join(root, name)
        with open(file_path, 'r', encoding='utf-8', errors = 'ignore') as file:
            lines = file.readlines()
            for line in lines:
                try: 
                    json_data = json.loads(line.strip())
                    doc = create_document(json_data)
                    writer.addDocument(doc)
                except Exception as e: 
                    continue
writer.close()
index_directory.close()

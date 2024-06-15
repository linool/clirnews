import lucene
import os
import json
from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexOptions, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory

try:
    lucene.initVM(lucene.CLASSPATH, maxheap='3g')
except Exception as e:
    print(e)

def create_document(json_data):
    id_field_type = FieldType()
    id_field_type.setStored(True)
    id_field_type.setTokenized(False)
    id_field_type.setIndexOptions(IndexOptions.DOCS)
    url_field_type = FieldType()
    url_field_type.setStored(True)
    url_field_type.setTokenized(False)
    url_field_type.setIndexOptions(IndexOptions.DOCS)
    title_field_type = FieldType()
    title_field_type.setStored(True)
    title_field_type.setTokenized(True)
    title_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
    text_field_type = FieldType()
    text_field_type.setStored(True)
    text_field_type.setTokenized(True)
    text_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    doc = Document()
    doc.add(Field('id', json_data['id'], id_field_type))
    doc.add(Field('url', json_data['url'], url_field_type))
    doc.add(Field('title', json_data['title'], title_field_type))
    doc.add(Field('titletext', json_data['title'] + ' \n ' + json_data['text'], text_field_type))
    return doc

docs_folder = '/path/mt_neuclir1/english'
index_directory = FSDirectory.open(Paths.get("/path/index_english"))
config = IndexWriterConfig(EnglishAnalyzer())
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
                    print("Error processing JSON:", e)
                    continue
writer.close()
index_directory.close()

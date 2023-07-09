# LLM 1
# Convert the user query into company names, product names and topics discussed

import os
from langchain.llms import VertexAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.llms import VertexAI
import re
from langchain.document_loaders import TextLoader
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain.chat_models import ChatGooglePalm, ChatVertexAI
from collections import defaultdict

with open('user_request.txt','r') as f:
    user_request = f.read()


tandc_list = ""
for comp in os.listdir("T&C_DatasetMD"):
    tandc_list+=comp
    tandc_list+=": "
    for prod in os.listdir(f"T&C_DatasetMD\{comp}"):
        prod_name = prod.split(".")[0]
        tandc_list+=prod_name+", "
    tandc_list=tandc_list[:-2]
    tandc_list+="\n\n"

company_names = ResponseSchema(
    name="Company_Names",
    description="Name of the companies"
)

product_names = ResponseSchema(
    name="Product_Names",
    description="Name of the products"
)
topics = ResponseSchema(
    name="topics",
    description="Keywords or topics discussed"
)


response_schema = [
    company_names,
    product_names,
    topics
]

output_parse = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parse.get_format_instructions()


llm1_template = """
You have the company names and their products {tandc_list}. Here each line is formatted as "Company_Name": Name of the products separated by commas.\n

Based ONLY on the list that you are provided above, respond to the user query and output the company names, product names and the topics, discussed in the query.\n

For topics, DON'T INCLUDE the company name or product names. \n

If company name isn't in the list, then don't include it in the output\n

If product name is not mentioned explicitly, then provide the most relevant product from the list that you are provided.\n

User request: "{user_request}"
{format_instructions}
"""

llm_1_prompt_template =PromptTemplate(
    input_variables=["tandc_list","user_request","format_instructions"],
    template = llm1_template
)
llm1_prompt =  llm_1_prompt_template.format(
    tandc_list=tandc_list,
    user_request=user_request,
    format_instructions=format_instructions
)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "participant-sa-15-ghc-011.json"
llm1 = VertexAI(temperature=0.0)


output = llm1.predict(llm1_prompt)

llm1_output_dict = output_parse.parse(output)
for key,val in llm1_output_dict.items():
    if not isinstance(val,list):
        llm1_output_dict[key] = val.split(",")

print(llm1_output_dict)

# Vector Store

#Adding \n at end of each sentence and \n\n at end of each paragraph
def add_line_breaks(text):
    sentence_splits = text.split(".")
    sentence_with_delimiter = ".\n".join(sentence_splits)
    #If number of whitespaces is more than 2, then it is another paragraph
    sentence_with_delimiter = re.sub(r' {2,}', '\n\n', sentence_with_delimiter)
    return sentence_with_delimiter

documents = []
metadata = []

for comp in tqdm(os.listdir("T&C_DatasetMD")):
    for prod in os.listdir(f"T&C_DatasetMD\{comp}"):
        # if prod == "Apple Fitness+.docx":continue
        loader = TextLoader(f"T&C_DatasetMD\{comp}\{prod}",autodetect_encoding=True)
        doc = loader.load()
        text = add_line_breaks(doc[0].page_content)
        documents.append(text)
        metadata.append({"company":comp,"product":prod.split(".")[0]})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=250)

# Split each element in the list
split_list = [text_splitter.split_text(element) for element in documents]

splitted_docs = []
splitted_metadata = []

for idx,docs in enumerate(split_list):
    curr_metadata = metadata[idx]
    if isinstance(docs,list):
        for doc in docs:
            splitted_docs.append(doc)
            splitted_metadata.append(curr_metadata)
    else:
        splitted_docs.append(docs)
        splitted_metadata.append(curr_metadata)

chroma_client = chromadb.Client()
collection_name = "TandC-project"

if len(chroma_client.list_collections()) > 0 and collection_name in [
    chroma_client.list_collections()[0].name
]:
    chroma_client.delete_collection(name=collection_name)
else:
    print(f"Creating collection: '{collection_name}'")
    collection = chroma_client.create_collection(name=collection_name)

print("Building the vector database")
collection.add(
    documents=splitted_docs,
    metadatas=splitted_metadata,
    ids=[f"id{i}" for i in range(1,len(splitted_docs)+1)]
)

def get_where_clause(output_dict):
    where_list = []
    for product_names in output_dict['Product_Names']:
        where_list.append({"product":product_names})
    return where_list

where_list = get_where_clause(llm1_output_dict)

if len(where_list)<=1:

    query_results = collection.query(
        query_texts=user_request,
        n_results=20,
        where=where_list[0]
    )
else:
    query_results = collection.query(
    query_texts=user_request,
    n_results=20,
    where={
            "$or":where_list
            }
)

relevant_dict = defaultdict(list)

for doc,meta in zip(query_results['documents'][0],query_results['metadatas'][0]):
    # print(meta,doc)
    relevant_dict[meta['product']].append(doc)

relevant_sentences = ""

for key,value in relevant_dict.items():
    value = " ".join(value)
    relevant_sentences+=f"Relevant document for {key}: "
    relevant_sentences+= value
    relevant_sentences+="\n\n"

print(relevant_sentences)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "participant-sa-15-ghc-011.json"

llm2_template = """
You are a legal expert with a strong knowledge of companies terms and conditions. Base your answer only on the following relevant documents: \n
{relevant_documents} \n\n

Answer the user questions {user_query}\n

DON'T make up any information, and answer only from the relevant documents.\n
"""

llm2_prompt_template = PromptTemplate(
    input_variables=["relevant_documents","user_query"],
    template=llm2_template
)

llm2_prompt =  llm2_prompt_template.format(
    user_query = "",
    relevant_documents=relevant_sentences
)


llm_2 = ChatVertexAI(temperature=0.5,max_output_tokens=1024)

output = llm_2.predict(llm2_prompt)
print(output)
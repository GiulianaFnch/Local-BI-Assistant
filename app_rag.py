from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------
# PASSOS 1 & 2: Carregar o documento e dividir em fragmentos (Chunks)
# ---------------------------------------------------------
print("1. A carregar o documento da empresa...")
document_path = "documentos/dados_empresa.txt"
with open(document_path, encoding="utf-8") as document_file:
    documents = [
        Document(
            page_content=document_file.read(),
            metadata={"source": document_path},
        )
    ]

# Dividimos o texto em pedaços pequenos para facilitar a pesquisa vetorial
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# ---------------------------------------------------------
# PASSO 3: Criar Embeddings e guardar no Vector Database (ChromaDB)
# ---------------------------------------------------------
print("2. A gerar Embeddings e a povoar a Vector Database local...")
# Modelo leve de embeddings do Hugging Face para português/multilíngue
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Guardamos os vetores em memória usando o ChromaDB
vector_store = Chroma.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Recupera os 2 trechos mais relevantes

# ---------------------------------------------------------
# PASSO 4: Carregar o LLM local do Hugging Face
# ---------------------------------------------------------
print("3. A carregar o LLM local (TinyLlama)...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Criar a pipeline de geração de texto
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.1, # Temperatura baixa para evitar alucinações
    repetition_penalty=1.1
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# ---------------------------------------------------------
# PASSO 5: Criar a cadeia RAG e testar a pergunta
# ---------------------------------------------------------
print("4. A inicializar o sistema de pergunta/resposta RAG...")
prompt = PromptTemplate.from_template(
    "Responde à pergunta exclusivamente com base no contexto abaixo. "
    "Se a resposta não estiver no contexto, diz que não sabes.\n\n"
    "Contexto:\n{context}\n\n"
    "Pergunta: {question}\n"
    "Resposta:"
)

def prepare_qa_inputs(inputs):
    question = inputs["question"]
    source_documents = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_documents)
    return {
        **inputs,
        "context": context,
        "source_documents": source_documents,
    }

answer_chain = prompt | local_llm | StrOutputParser()
qa_chain = RunnableLambda(prepare_qa_inputs) | RunnablePassthrough.assign(
    result=answer_chain
)

# Teste prático de pergunta em linguagem natural
pergunta = "Quando posso marcar minhas férias?"
print(f"\n--- Pergunta: {pergunta} ---")

resposta = qa_chain.invoke({"question": pergunta})

print("\n--- Resposta Gerada ---")
print(resposta["result"])

print("\n--- Fontes Utilizadas ---")
for doc in resposta["source_documents"]:
    print(f"- {doc.page_content}")
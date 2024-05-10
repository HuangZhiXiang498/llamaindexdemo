from dotenv import load_dotenv

load_dotenv()

import os
import logging
import threading

from llama_index.core.storage import StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from app.settings import init_settings
from app.engine.loaders import get_documents
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 假设 documents 是一个列表，我们将其分成多个批次以供多线程处理
def process_batch(batch):
    # 这里是你的 VectorStoreIndex.from_documents 函数调用
    # 你需要修改这个函数调用，使其只处理一个batch的数据
    store = MilvusVectorStore(
        uri=os.environ["MILVUS_ADDRESS"],
        user=os.getenv("MILVUS_USERNAME"),
        password=os.getenv("MILVUS_PASSWORD"),
        collection_name=os.getenv("MILVUS_COLLECTION"),
        dim=int(os.getenv("EMBEDDING_DIM")),
    )
    print(os.getenv("MILVUS_COLLECTION"))
    storage_context = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex.from_documents(
        batch,
        storage_context=storage_context,
        show_progress=True,  # 在多线程中，通常不需要显示进度条
    )
    # 处理完毕后的逻辑（如果有的话）
    logger.info("Processed a batch")
def generate_datasource():
    init_settings()
    logger.info("Creating new index")
    documents = get_documents()  # 假设这是获取文档数据的函数

    # 确定要使用的线程数
    num_threads = int(os.getenv("num_threads"))

    # 将文档分成num_threads个批次
    batch_size = len(documents) // num_threads
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    # 使用线程池执行
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 将每个批次的处理任务提交给线程池
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

        # 等待所有任务完成
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                # 这里可以获取异常信息，如果有必要的话
                result = future.result()
            except Exception as exc:
                print(f'Batch generated an exception: {exc}')
            else:
                # 打印成功信息 获取当前线程id
                print(f'Batch processed: {threading.get_ident()}')

    logger.info(f"Successfully created embeddings in the Milvus")



if __name__ == "__main__":
    generate_datasource()

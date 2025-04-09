from vendor.vector_store import DataToVectorStoreProcessor
import logging
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(
    filename=os.path.normpath(os.path.join(script_dir, '..', 'log', 'vector_store_processing.log')),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('faiss').setLevel(logging.WARNING)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    index_path = os.path.normpath(os.path.join(project_dir, 'data', 'l2'))

    new_file_path = os.path.join(project_dir, 'data', 'bcr_e_tombados.csv')

    csv_config = {
        'file_path': new_file_path
    }
    chunk_size = 750
    chunk_overlap = 150

    processor = DataToVectorStoreProcessor(source_type="csv",
                                           source_config=csv_config,
                                           chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap,
                                           distance_strategy='l2',
                                           index_path=index_path)

    processor.manage_vector_store("add_new")

from vendor.vector_store import DataToVectorStoreProcessor
import os


script_dir = os.path.dirname(os.path.realpath(__file__))
test_file_path = os.path.normpath(os.path.join(script_dir, 'test_file.json'))

json_config = {
    'file_path': test_file_path,
    'jq_schema': '.[] | {name: .name, post_title: .post_title, post_content: .post_content, taxonomy_name_values: .taxonomy_name_values, post_date: .post_date}'
}

processor = DataToVectorStoreProcessor(
    source_type="json",
    source_config=json_config,
    chunk_size=500,
    chunk_overlap=100
)

processor.process()

print("Artigos processados e armazenados com sucesso na vector store!")

from vendor.vector_store import DataToVectorStoreProcessor


json_config = {
    'file_path': 'test_file.json',
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

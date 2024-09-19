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

    query = """
    with taxonomy_name AS (
        SELECT 'Classificação da organização' AS taxonomy_name, 237 AS taxonomy_suffix
        UNION ALL
        SELECT 'Estados', 12453
        UNION ALL
        SELECT 'Estados e Municípios brasileiros', 2767
        UNION ALL
        SELECT 'Municípios', 12455
        UNION ALL
        SELECT 'Processo institucional', 15122
        UNION ALL
        SELECT 'Processo institucional vinculado', 250
        UNION ALL
        SELECT 'Unidade do IPHAN', 13896
        UNION ALL
        SELECT 'Unidade Federativa', 185
        UNION ALL
        SELECT 'Unidades', 13443
    ),
    collections_name as (
        SELECT 'Ações' AS name, 819 AS post_type_suffix
        UNION ALL
        SELECT 'Ações Vigentes', 13339
        UNION ALL
        SELECT 'Educação Patrimonial', 16893
        UNION ALL
        SELECT 'Formação de agenda compartilhada no territorio', 14649
        UNION ALL
        SELECT 'Integração do IPHAN em instâncias participativas', 14614
        UNION ALL
        SELECT 'Organizações', 8
        UNION ALL
        SELECT 'Relação do Iphan com as organizações', 15441
    ),
    collections_value as (
        -- select *
        select term_taxonomy_id, SUBSTRING_INDEX(SUBSTRING_INDEX(post_type, '_', -2), '_', 1) AS post_type_suffix, post_title, post_content, post_date
        from wp_posts
        left join wp_term_relationships on wp_posts.ID = wp_term_relationships.object_id
        where wp_posts.post_type = 'tnc_col_819_item' and wp_posts.post_status = 'publish' -- organizacoes
    ),
    collections_value_name as (
        select 
            term_taxonomy_id, 
            CONCAT('Coleção: ', name) as name,
            CONCAT('Título: ', post_title) as post_title, 
            CONCAT('Descrição: ', post_content) as post_content, 
            CONCAT('Data de postagem: ', post_date) as post_date
        from collections_value
        inner join collections_name using (post_type_suffix)
    ), 
    term_value as (
        -- select *
        select term_taxonomy_id, SUBSTRING(wp_term_taxonomy.taxonomy, 9) AS taxonomy_suffix
    ,name as value, parent
        from wp_term_taxonomy
        left join wp_terms USING (term_id)
        where wp_term_taxonomy.taxonomy like 'tnc_tax_%'
    ),
    term_value_parent as (
        select 
            b1.term_taxonomy_id, 
            b1.taxonomy_suffix,
            CONCAT(b1.value, CASE WHEN b2.value IS NOT NULL THEN CONCAT(' - ', b2.value) ELSE '' END) AS name_parent
        from term_value b1
        left join term_value b2 on b1.parent = b2.term_taxonomy_id
    ),
    collections_term as (
        select 
            name, 
            post_title, 
            post_content, 
            term_taxonomy_id,
            taxonomy_suffix,
            GROUP_CONCAT(name_parent ORDER BY name_parent SEPARATOR ', ') AS name_parent_list,
            post_date
        from collections_value_name
        left join term_value_parent using (term_taxonomy_id)     
        group by name, post_title, post_content, term_taxonomy_id, post_date, taxonomy_suffix
    ),
    final as (
        select 
            name, 
            post_title, 
            post_content, 
            GROUP_CONCAT(COALESCE(taxonomy_name.taxonomy_name, ''), ': ', collections_term.name_parent_list) AS taxonomy_name_values,
            post_date
        from collections_term
        inner join taxonomy_name USING (taxonomy_suffix)
        group by name, post_title, post_content, post_date
    )
    select *
    from final
    """

    db_config = {
        'query': query,
        'host': 'localhost',
        'port': 3306,
        'user': 'user_leitura',
        'password': 'senha@123',
        'database': 'dbwordpress'
    }

    save_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'sql_MAX_INNER_PRODUCT_index'))
    
    processor = DataToVectorStoreProcessor(source_type="sql",
                                           source_config=db_config,
                                           chunk_size=600,
                                           chunk_overlap=100,
                                           distance_strategy='l2')
    processor.process()

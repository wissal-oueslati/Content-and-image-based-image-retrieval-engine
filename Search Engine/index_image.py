from elasticsearch import Elasticsearch
import csv


def index_feature_vectors_to_elasticsearch(csv_file_path):
    es = Elasticsearch(hosts=["http://localhost:9200"])

    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header row

        for row in reader:
            image_id, vector_str = row[0], row[1]
            vector = [float(i) for i in vector_str.split(',')]

            body = {
                "image_id": image_id,
                "image_vector": vector
            }

            es.index(index="image_vector_index_final_version", body=body)

    print("Indexing completed!")


if __name__ == "__main__":
    csv_file_path = 'extracted_feature.csv'
    index_feature_vectors_to_elasticsearch(csv_file_path)
# Content and Text Based Image Retrieval Search Engine
## Context
The Content and Text-Based Image Retrieval Search Engine is an innovative tool designed to simplify the process of finding images based on either visual content or textual descriptions. It covers over 2 million annotated images from the Yahoo Flickr Dataset. This search engine enables users to search using text or images.

![image](https://github.com/mariem-mezghani/Search-Engine/assets/118765518/f3f8cd37-1e99-4e2f-87d5-8c6e1defbe61)

## Getting started
### Prerequisites
First you should install:

* ElasticSearch 8.10.4
https://www.elastic.co/fr/downloads/past-releases/elasticsearch-8-10-4

* Logstash 8.10.4
https://www.elastic.co/fr/downloads/past-releases/logstash-8-10-4

### Installation
These are the steps you should follow in order to run this project:
1. Clone the repository.
2. Download the file **photo_metadata.csv** using this link:
https://yahooresearch.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images
4. Create a new index by executing **create_index.py** file

**NB:** You can check if you added the index successfully using this link:
http://localhost:9200/_cat/indices

6. Create the file **extracted_feature.csv** by executing the **Feature_extarction.py** file to extract features from images
7. Run this command:
```
streamlit run app.py
```

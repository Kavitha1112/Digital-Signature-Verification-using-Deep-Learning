# import sys
# print(sys.executable)
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import pymongo
import os

load_dotenv()
MONGODB_CONNECTION = os.getenv("MONGODB")

def connect_to_mongo():
    try:
        CONNECTION_STRING = MONGODB_CONNECTION
        
        client = MongoClient(CONNECTION_STRING) 
        db = client['Signature_Matching']
        print("Successfully Connected to MongoDB")
        return db
    except errors.ConnectionError as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

    
def data_store(db, documents):
    try:
        collection = db['tbl_signatures']
        
        # Check if index exists on _id and person_name
        # if 'person_name_1__id_1' not in collection.index_information():
            # db['tbl_signatures'].create_index([('person_name', pymongo.ASCENDING), ('_id', pymongo.ASCENDING)])
        if 'person_name_1' not in collection.index_information():
            collection.create_index([('person_name', pymongo.ASCENDING)])
            collection.create_index([('person_name', pymongo.TEXT)])
            print("Created index on 'person_name'")
        
        # Insert documents
        results = collection.insert_many(documents, ordered=False)  
        print("The images have been successfully saved.")
        return True
    except errors.BulkWriteError as bwe:
        print(f"Bulk write error: {bwe.details}")
        return False
    except errors.PyMongoError as e:
        print(f"Error while saving images in MongoDB: {e}")
        return False


def fetch_signatures(db, person_name):
    try:
        collection = db['tbl_signatures']
        documents = list(collection.find({'person_name': person_name}, {'signature': 1, '_id': 1}))
  
        if documents:
            print(f"The signatures of {person_name} have been fetched successfully.")
            return documents 
        else:
            print(f"No images found for {person_name}")
            return None
    except errors.PyMongoError as e:
        print(f"Error when fetching images from MongoDB: {e}")
        return None


def get_unique_person_names(db):
    try:
        collection = db['tbl_signatures']
        person_names = collection.distinct('person_name')
        return person_names
    except errors.PyMongoError as e:
        print(f"Error fetching unique person names: {e}")
        return None


def search_person_names(db, query, limit=10): 
    try:
        if not query:
            return []  # Return an empty list for an empty query

        collection = db['tbl_signatures']

        # Use regex filtering to match `person_name` and then fetch distinct names
        pipeline = [
            {
                "$match": {
                    "person_name": {"$regex": query, "$options": "i"}  # Case-insensitive regex match
                }
            },
            {
                "$group": {
                    "_id": "$person_name"  # Group by person_name to get unique values
                }
            },
            {
                "$sort": {"_id": 1}  # Sort by name alphabetically (optional)
            },
            {
                "$limit": limit  
            }
        ]

        results = collection.aggregate(pipeline)
        return [result["_id"] for result in results]
    except Exception as e:
        print(f"Error fetching unique person names: {e}")
        return []

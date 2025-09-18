import weaviate

client = weaviate.connect_to_local()


# List all collections
collections = client.collections.list_all()
print("Available collections:", collections.keys())

# Replace 'YourCollectionName' with the name of the collection you want to inspect
collection_name = "DocumentChildChunk"

if collection_name in collections:
    my_collection = client.collections.get(collection_name)

    # Use the iterator to get all objects in the collection
    for item in my_collection.iterator():
        print(item.properties)
else:
    print(f"Collection '{collection_name}' not found.")

client.close()

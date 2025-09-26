import weaviate
from weaviate.classes.query import MetadataQuery

# ==========================
# ===== Connect to Weaviate =====
# ==========================

try:
    client = weaviate.connect_to_local()
    print("Successfully connected to Weaviate.")

    # ===================================
    # ===== Specify Your Collection =====
    # ===================================

    # !!! IMPORTANT !!!
    # Replace "YourCollectionName" with the actual name of your collection in Weaviate.
    collection_name = "DocumentChildChunk"

    # Get the collection object
    my_collection = client.collections.get(collection_name)
    print(f"Successfully got the '{collection_name}' collection.")

    # ==================================================
    # ===== Fetch Objects and Their Vectors =====
    # ==================================================

    # Fetch a limited number of objects from the collection
    # The `include_vector=True` argument is essential to retrieve the vector.
    response = my_collection.query.fetch_objects(limit=5, include_vector=True)

    # ==============================================
    # ===== Print the Retrieved Vectors =====
    # ==============================================

    print("\n--- Retrieved Objects and their Vectors ---")
    if len(response.objects) > 0:
        for i, obj in enumerate(response.objects):
            print(f"\n--- Object {i+1} ---")

            # Print some properties of the object to identify it
            print(f"Properties: {obj.properties}")

            # Print the vector
            # The vector is accessed via the .vector attribute of the object
            if obj.vector:
                print(obj.vector)
                print(f"Vector (first 10 dimensions): {obj.vector['default'][:10]}...")
                print(f"Vector dimension: {len(obj.vector)}")
            else:
                print("Vector not found for this object.")
    else:
        print("No objects found in the collection.")

finally:
    # Close the connection to Weaviate
    if "client" in locals() and client.is_connected():
        client.close()
        print("\nSuccessfully closed the connection to Weaviate.")

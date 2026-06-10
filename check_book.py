import pickle
import os

book_file = 'opening_book.pkl'
if os.path.exists(book_file):
    file_size = os.path.getsize(book_file)
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    
    with open(book_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Opening book entries: {len(data)}")
    if data:
        print(f"Sample FEN: {list(data.keys())[0]}")
        print(f"Sample move: {data[list(data.keys())[0]]}")
    else:
        print("Opening book is EMPTY - generation didn't complete!")
else:
    print("opening_book.pkl not found")

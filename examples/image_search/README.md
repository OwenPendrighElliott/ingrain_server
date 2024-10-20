# Image Search Demo

To get started with the image search demo, you can use the optimised Python Client:

```bash
pip install ingrain
```

Install the required packages:

```bash
pip install hnswlib numpy pillow flask tqdm
```

Then place your images in a folder called `images` and run the following code to index the data:

NOTE: There may be a delay and slow start as the workers configure their preprocessing configs.
```python
python index_data.py
```

After the data has been indexed, you can run the following code to start the server to play with the search:

```python
python app.py
```

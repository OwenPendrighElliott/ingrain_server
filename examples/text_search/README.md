# Text Search Demo

To get started with the image search demo, you can use the optimised Python Client:

```bash
pip install ingrain
```

Install the required packages:

```bash
pip install flask tqdm
```

Run a HNSWLib Server:

```bash
docker run -p 8685:8685 owenpelliott/hnswlib_server
```

Download the scidocs data:

```bash
bash get_data.sh
```

Then run the following code to index the data:

NOTE: There may be a delay and slow start as the workers configure their preprocessing configs.
```bash
python index_data.py
```

After the data has been indexed, you can run the following code to start the server to play with the search:

```bash
python app.py
```

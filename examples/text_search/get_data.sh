echo "Downloading dataset"
mkdir -p data
mkdir -p data_zips

if [ ! -d "data/scidocs" ]; then
    echo "Downloading scidocs"
    wget -O data_zips/scidocs.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip
    unzip data_zips/scidocs.zip -d data/scidocs
else
    echo "scidocs already exists"
fi
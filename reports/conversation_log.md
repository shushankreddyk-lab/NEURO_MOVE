# Conversation Log & Project Updates

## Update - BCI Competition IV 2a Integration
* Added global UI toggle for selecting between PhysioNet EEGMMIDB (.edf) and BCI Competition IV 2a (.gdf) datasets in the Live Training tab.
* Updated `train_master.py` to correctly extract, dynamically pad (to 656 samples), and build models for the BCI dataset's 22-channel configuration.
* Added Literature Benchmark metrics specifically for the BCI Competition dataset in the Technical Details tab, mirroring the performance of the baseline CNN-LSTM and MiniRocket pipelines.
* Enhanced the Live Inference component to seamlessly ingest `.gdf` files alongside `.edf` files, dynamically switching target channels and sampling logic (160Hz resampling) based on the user's selected architecture.

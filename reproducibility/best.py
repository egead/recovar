import os, pandas as pd, json
with open('settings.json') as f: settings = json.load(f)
for root, dirs, files in os.walk(settings['TRAINED_MODELS_DIR']):
	if 'history.csv' in files:
		df = pd.read_csv(os.path.join(root, 'history.csv'))
		rel = os.path.relpath(root, settings['TRAINED_MODELS_DIR'])
		print(f'{rel} -> best epoch: {df["val_loss"].idxmin()}')

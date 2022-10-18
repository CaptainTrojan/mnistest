import pandas as pd
import plotly.express as px
import os

ROOT = "results"
#rows = {'batch_size': [], '10_epoch_loss': [], 'convergence_loss': [], 'convergence_time': []}

#for dirname in os.listdir(ROOT):
#	for file in os.listdir(os.path.join(ROOT, dirname)):
#		with open(os.path.join(ROOT, dirname, file)) as f:
#			nums = f.read().strip().split(',')
#			rows['batch_size'].append(int(dirname))
#			rows['10_epoch_loss'].append(float(nums[0]))
#			rows['convergence_loss'].append(float(nums[1]))
#			rows['convergence_time'].append(float(nums[2]))

#df = pd.DataFrame(rows)
#df.to_csv('results.csv', index=False)

df = pd.read_csv('results.csv')

fig = px.box(df, x='batch_size', y='10_epoch_loss')
fig.write_image('output1.pdf')

fig = px.box(df, x='batch_size', y='convergence_loss')
fig.write_image('output2.pdf')

fig = px.box(df, x='batch_size', y='convergence_time')
fig.write_image('output3.pdf')

# Speed decoupling — position decoding from neural activity, speed, or both

Five position decoders, one per column group, identical in everything but their
input, so every comparison is paired:

- MLP (Optuna, 20 trials; 5 trial-wise outer folds × 1 validation hold-out) 
- Same kept bins (post-context tail removed at 515° ≈ 62.9 cm, Gaussian smoothing σ = 200 ms, no history window) 
- Same trial folds

Experiments:

- **Neural Act.** (Batch 007): smoothed spike counts of the kept neurons in the
  current 200 ms bin, plus the trial's context one-hot (rewarded / unrewarded).
- **Speed (window)** (Batch 011): no neural data. The speed input is a window of five raw
  binned speed values, from two bins before to two bins after the current one
  (t − 400 ms … t + 400 ms), taken on the continuous recording before any bin is
  removed — the same convention as the Gaussian smoothing of the neural input.
  The same context one-hot is appended (7 inputs).
- **Neural Act. + Speed (window)** (Batch 012): the Neural Act. features (without
  their context columns) plus the 7 Speed (window) inputs.
- **Speed (smoothed)** (Batch 013): no neural data. One speed value: the continuous
  raw speed series smoothed with exactly the Gaussian applied to the spike counts
  (σ = 200 ms, same function), sampled at the current bin. Speed and neurons then
  share the same temporal transformation, whereas the window also exposes
  acceleration. The same context one-hot is appended (3 inputs).
- **Neural Act. + Speed (smoothed)** (Batch 014): the Neural Act. features (without
  their context columns) plus the 3 Speed (smoothed) inputs.

Values are medians over the 5 outer test folds: R² and mean absolute error (MAE,
cm). Summary rows are medians of the per-recording values.

<table>
<thead>
<tr><th rowspan="2">Recording</th>
<th colspan="2">Neural Act.</th>
<th colspan="2">Speed (window)</th>
<th colspan="2">Neural Act. + Speed (window)</th>
<th colspan="2">Speed (smoothed)</th>
<th colspan="2">Neural Act. + Speed (smoothed)</th>
</tr>
<tr><th>R²</th><th>MAE (cm)</th><th>R²</th><th>MAE (cm)</th><th>R²</th><th>MAE (cm)</th><th>R²</th><th>MAE (cm)</th><th>R²</th><th>MAE (cm)</th></tr>
</thead>
<tbody>
<tr><td>Pir-L7-Naive</td><td>0.110</td><td>14.9</td><td>0.096</td><td>16.2</td><td>0.141</td><td>14.4</td><td>0.045</td><td>17.2</td><td>0.143</td><td>14.8</td></tr>
<tr><td>Pir-N53-Naive</td><td>0.178</td><td>14.5</td><td>0.101</td><td>14.9</td><td>0.194</td><td>14.3</td><td>0.086</td><td>15.2</td><td>0.196</td><td>14.3</td></tr>
<tr><td>Pir-L3-Expert</td><td>0.460</td><td>10.3</td><td>0.433</td><td>11.6</td><td>0.614</td><td>8.1</td><td>0.262</td><td>13.4</td><td>0.605</td><td>8.7</td></tr>
<tr><td>Pir-L5-Expert</td><td>0.499</td><td>11.0</td><td>0.330</td><td>12.4</td><td>0.622</td><td>8.8</td><td>0.308</td><td>13.3</td><td>0.599</td><td>9.1</td></tr>
<tr><td>Pir-N31-Expert</td><td>0.571</td><td>10.1</td><td>0.465</td><td>11.4</td><td>0.693</td><td>8.1</td><td>0.372</td><td>12.5</td><td>0.673</td><td>8.7</td></tr>
<tr><td>Pir-N59-Expert</td><td>0.215</td><td>15.6</td><td>0.239</td><td>15.2</td><td>0.314</td><td>13.5</td><td>0.189</td><td>15.9</td><td>0.292</td><td>14.0</td></tr>
<tr><td>CA3-S30-Naive</td><td>0.162</td><td>15.9</td><td>0.243</td><td>14.8</td><td>0.272</td><td>14.6</td><td>0.184</td><td>15.8</td><td>0.230</td><td>15.0</td></tr>
<tr><td>CA3-S31-Naive</td><td>0.268</td><td>14.2</td><td>0.179</td><td>15.6</td><td>0.378</td><td>12.6</td><td>0.149</td><td>16.0</td><td>0.360</td><td>12.9</td></tr>
<tr><td>CA3-S17-Expert</td><td>0.295</td><td>12.4</td><td>0.300</td><td>12.3</td><td>0.441</td><td>10.6</td><td>0.193</td><td>14.4</td><td>0.377</td><td>11.5</td></tr>
<tr><td>CA3-S18-Expert</td><td>0.442</td><td>11.7</td><td>0.477</td><td>11.0</td><td>0.593</td><td>9.2</td><td>0.426</td><td>12.0</td><td>0.577</td><td>9.5</td></tr>
<tr><td>DG-S24-Naive</td><td>0.314</td><td>12.8</td><td>0.112</td><td>15.6</td><td>0.400</td><td>11.5</td><td>0.080</td><td>16.2</td><td>0.375</td><td>12.1</td></tr>
<tr><td>DG-S38-Naive</td><td>0.101</td><td>14.5</td><td>0.116</td><td>15.4</td><td>0.149</td><td>14.0</td><td>0.098</td><td>15.8</td><td>0.135</td><td>14.7</td></tr>
<tr><td>DG-S39-Naive</td><td>0.074</td><td>17.4</td><td>0.169</td><td>15.5</td><td>0.190</td><td>15.0</td><td>0.128</td><td>16.1</td><td>0.139</td><td>15.8</td></tr>
<tr><td>DG-S41-Naive</td><td>0.206</td><td>15.3</td><td>0.242</td><td>15.0</td><td>0.422</td><td>12.0</td><td>0.195</td><td>15.5</td><td>0.321</td><td>13.3</td></tr>
<tr><td>DG-S17-Expert</td><td>0.369</td><td>11.4</td><td>0.314</td><td>11.6</td><td>0.450</td><td>10.2</td><td>0.169</td><td>13.9</td><td>0.434</td><td>10.5</td></tr>
<tr><td>DG-S18-Expert</td><td>0.142</td><td>15.7</td><td>0.238</td><td>14.4</td><td>0.212</td><td>13.8</td><td>0.166</td><td>15.8</td><td>0.196</td><td>14.3</td></tr>
<tr><td>DG-S19-Expert</td><td>0.245</td><td>14.1</td><td>0.431</td><td>12.0</td><td>0.524</td><td>10.5</td><td>0.348</td><td>13.5</td><td>0.471</td><td>11.6</td></tr>
<tr><td><b>Piriform · Expert (n=4)</b></td><td><b>0.480</b></td><td><b>10.7</b></td><td><b>0.381</b></td><td><b>12.0</b></td><td><b>0.618</b></td><td><b>8.5</b></td><td><b>0.285</b></td><td><b>13.4</b></td><td><b>0.602</b></td><td><b>8.9</b></td></tr>
<tr><td><b>Piriform · Naive (n=2)</b></td><td><b>0.144</b></td><td><b>14.7</b></td><td><b>0.099</b></td><td><b>15.6</b></td><td><b>0.167</b></td><td><b>14.4</b></td><td><b>0.065</b></td><td><b>16.2</b></td><td><b>0.169</b></td><td><b>14.5</b></td></tr>
<tr><td><b>Hippocampus · Expert (n=5)</b></td><td><b>0.295</b></td><td><b>12.4</b></td><td><b>0.314</b></td><td><b>12.0</b></td><td><b>0.450</b></td><td><b>10.5</b></td><td><b>0.193</b></td><td><b>13.9</b></td><td><b>0.434</b></td><td><b>11.5</b></td></tr>
<tr><td><b>Hippocampus · Naive (n=6)</b></td><td><b>0.184</b></td><td><b>14.9</b></td><td><b>0.174</b></td><td><b>15.5</b></td><td><b>0.325</b></td><td><b>13.3</b></td><td><b>0.138</b></td><td><b>15.9</b></td><td><b>0.276</b></td><td><b>14.0</b></td></tr>
<tr><td><b>All 17 recordings (n=17)</b></td><td><b>0.245</b></td><td><b>14.2</b></td><td><b>0.242</b></td><td><b>14.8</b></td><td><b>0.400</b></td><td><b>12.0</b></td><td><b>0.184</b></td><td><b>15.5</b></td><td><b>0.360</b></td><td><b>12.9</b></td></tr>
</tbody>
</table>

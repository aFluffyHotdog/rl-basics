A StableBaselines3 environment used to train 

### Getting started (Windows)

1. Create a venv with 
```
python3 -m venv .venv
```
2. Activate it with 
```
.venv\Scripts\activate
```
3. Tell the shell where Vivado is (this is an example path, please change accordingly)
```
$env:PATH += ";F:\Vivado\2022.1\bin"
```
4. Run with 
```
python3 .\training\train_xsim.py --data_dir sample_data_9_lanes --sim_dir C:\Users\macth\Documents\T8_Decomp_xsim\sim
```

### Getting started (Linux)

1. Create a venv with
```
python3 -m venv .venv
```
2. Activate it with
```
source .venv/bin/activate
```
3. Tell the shell where Vivado is (change this to your actual install path)
```
export PATH="$PATH:/opt/Xilinx/Vivado/2022.1/bin"
```
4. Run with
```
python3 ./training/train_xsim.py --data_dir ./sample_data_9_lanes --sim_dir /home/{PATH_TO_TB}/T8_Decomp_xsim/sim
```
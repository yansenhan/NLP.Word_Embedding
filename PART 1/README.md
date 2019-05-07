### Environments:
- Python 3.6
- tensorflow 1.13.1
- numpy 1.16.2

### How to Run the Code **(Recommend to run the code on IDE)**
You, firstly, need to make a directory 'data' in the 'PART I', and then download the data, [text8.zip](http://mattmahoney.net/dc/text8.zip), and unzip it to the directory 'data'.

#### Tree Structure 
```
├─PART 1
   ├─data
   └─src
       └─__pycache__
```
       
#### Run on IDE
- Firstly, you can directly run the code "all_model.py".
- If the "all_model.py" doesn't work, then you can run "src/skip_gram.py" and "src/glove.py"

#### Run on Console / Terminal
There are two ways to run the codes on the console / terminal:

- Firstly,
```
cd [your path of this document]
python all_models.py
```
or 
```
cd [your path of this document]
python3 all_models.py
```
- If it fails, let's run the code in the second way.
```
cd [your path of this document]/src
python skip_gram.py
python glove.py
```
or 
```
cd [your path of this document]/src
python3 skip_gram.py
python3 glove.py
```

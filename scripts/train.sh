export CUDA_VISIBLE_DEVICES=3

model_name=TimeLLM

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.00001
d_model=16
d_ff=32
train_epochs=100
patience=10
batch_size=128
root_data_path="/home/schaffen/Workspace/Project/CandleType_Forcasting/"
num_enc=22
llama_layers=32
d_model=32
d_ff=128

python -u run.py \
  --is_training 1 \
  --root_path  $root_data_path \
  --data_path data/XAUUSD_M5.csv \
  --model $model_name \
  --data XAUUSD \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs 

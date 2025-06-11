# export CUDA_VISIBLE_DEVICES=0

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
batch_size=4
root_data_path="/home/trangndp/projects/trading_bot/"
num_enc=22
llama_layers=6

python -u run.py \
  --is_training 1 \
  --root_path  $root_data_path \
  --data_path dataset/M5/XAUUSD_M5.csv \
  --model $model_name \
  --data XAUUSD \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs 

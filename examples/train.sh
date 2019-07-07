export GLUE_DIR=/data/share/zhanghaipeng/tre/datasets/data
export TASK_NAME=tacred

EXPR=25
BS=16
CUDA=2
LR=3e-5
EPOCH=4.0

CUDA_VISIBLE_DEVICES=$CUDA python tacred_run_classifier.py \
	--task_name $TASK_NAME \
	--do_train \
	--do_lower_case \
	--data_dir $GLUE_DIR/tacred/ \
	--max_seq_length 128 \
	--train_batch_size $BS \
	--learning_rate $LR \
	--num_train_epochs $EPOCH \
	--output_dir train/$EXPR/$TASK_NAME$EPOCH \
	--bert_model model/bert-base-uncased

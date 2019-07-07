#export GLUE_DIR=/data/share/zhanghaipeng/pytorch-pretrained-BERT/examples/general_ner_test
export GLUE_DIR=/data/share/zhanghaipeng/tre/datasets/data
export TASK_NAME=tacred

EXPR=23
BS=64
CUDA=0
EPOCH=3.0

CUDA_VISIBLE_DEVICES=$CUDA python tacred_run_classifier.py \
	--task_name $TASK_NAME \
	--do_test \
	--do_lower_case \
	--data_dir $GLUE_DIR/tacred/ \
	--max_seq_length 128 \
	--eval_batch_size $BS \
	--output_dir train/$EXPR/$TASK_NAME$EPOCH \
	--bert_model model/bert-large-uncased
